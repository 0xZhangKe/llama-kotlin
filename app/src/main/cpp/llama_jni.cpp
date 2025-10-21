// src/main/cpp/llama_jni.cpp
#include <jni.h>
#include <string>
#include <vector>
#include <deque>
#include <atomic>
#include <algorithm>
#include <cstring>

#include "llama.h"   // 确保 CMake 已把 llama.cpp 的头文件目录加入 include 路径

// ================== 全局状态（与 Kotlin 的 object 匹配） ==================
static llama_model*   g_model      = nullptr;
static llama_context* g_ctx        = nullptr;
static int            g_n_threads  = 4;
static int            g_n_ctx      = 4096;
static std::atomic<bool> g_cancel{false};

// 回调缓存
static jclass    g_CallbackCls   = nullptr;
static jmethodID g_OnTokenMethod = nullptr; // void onToken(String,int)
static jmethodID g_ShouldStop    = nullptr; // boolean shouldStop()

// --------- 工具 ----------
static void jthrow(JNIEnv* env, const char* msg) {
    jclass ex = env->FindClass("java/lang/RuntimeException");
    if (ex) env->ThrowNew(ex, msg);
}
static bool ensure_model(JNIEnv* env) {
    if (!g_model) { jthrow(env, "Model is null. Call Llama.load() first."); return false; }
    return true;
}
static bool ensure_ctx(JNIEnv* env) {
    if (!g_ctx) { jthrow(env, "Context is null. Call Llama.load() first."); return false; }
    return true;
}
static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}
// 单 token -> 文本
static std::string token_to_piece(llama_model* model, llama_token tok) {
    int need = llama_detokenize(model, &tok, 1, nullptr, 0, true);
    if (need <= 0) return "";
    std::string out; out.resize(need);
    int wrote = llama_token_to_piece(model, tok, out.data(), (int)out.size(), /*special=*/true);
    if (wrote < 0) return "";
    if (wrote < need) out.resize(wrote);
    return out;
}
// 多 token 简易 detok
static std::string detok(llama_model* model, const std::vector<llama_token>& toks) {
    std::string s; s.reserve(toks.size() * 4);
    for (auto t : toks) s += token_to_piece(model, t);
    return s;
}

// ================== JNI 生命周期 ==================
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    // 缓存回调方法（存在则支持流式）
    jclass cb = env->FindClass("com/zhangke/llama/Llama$TokenCallback");
    if (cb) {
        g_CallbackCls   = (jclass) env->NewGlobalRef(cb);
        g_OnTokenMethod = env->GetMethodID(g_CallbackCls, "onToken", "(Ljava/lang/String;I)V");
        g_ShouldStop    = env->GetMethodID(g_CallbackCls, "shouldStop", "()Z");
    }

    llama_backend_init(); // llama 全局初始化
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) == JNI_OK) {
        if (g_CallbackCls) { env->DeleteGlobalRef(g_CallbackCls); g_CallbackCls = nullptr; }
    }
    if (g_ctx)   { llama_free(g_ctx);   g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }
    llama_backend_free();
}

// ================== Llama.load / unload ==================
extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeLoadModel(
        JNIEnv* env, jclass,
        jstring jpath, jint n_ctx, jint n_gpu_layers, jint n_threads,
        jint seed, jboolean use_mmap, jboolean use_mlock) {

    // 允许重复 load：先清理旧的
    if (g_ctx)   { llama_free(g_ctx);   g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }

    const char* cpath = env->GetStringUTFChars(jpath, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_mmap     = (bool)use_mmap;
    mparams.use_mlock    = (bool)use_mlock;

    g_model = llama_load_model_from_file(cpath, mparams);
    env->ReleaseStringUTFChars(jpath, cpath);
    if (!g_model) { jthrow(env, "llama_load_model_from_file failed"); return; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.seed       = (seed < 0 ? LLAMA_DEFAULT_SEED : seed);
    cparams.embeddings = true; // 需要 embeddings

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        llama_free_model(g_model); g_model = nullptr;
        jthrow(env, "llama_new_context_with_model failed"); return;
    }

    g_n_threads = std::max(1, (int)n_threads);
    g_n_ctx     = n_ctx;
    g_cancel.store(false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeFreeModel(JNIEnv*, jclass) {
    if (g_ctx)   { llama_free(g_ctx);   g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }
    g_cancel.store(false);
}

// ================== 基础工具 ==================
extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeReset(JNIEnv* env, jclass) {
    if (!ensure_ctx(env)) return;
    llama_kv_cache_clear(g_ctx);
    g_cancel.store(false);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_zhangke_llama_Llama_nativeVocabSize(JNIEnv* env, jclass) {
    if (!ensure_model(env)) return 0;
    return (jint)llama_n_vocab(g_model);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_zhangke_llama_Llama_nativeTokenize(
        JNIEnv* env, jclass, jstring jtext, jboolean add_bos) {
    if (!ensure_model(env)) return nullptr;
    const char* ctext = env->GetStringUTFChars(jtext, nullptr);

    int cap = (int)strlen(ctext) + 8; // 预估容量
    std::vector<llama_token> ids(cap);
    int n = llama_tokenize(g_model, ctext, ids.data(), cap, (bool)add_bos);
    env->ReleaseStringUTFChars(jtext, ctext);
    if (n < 0) { jthrow(env, "llama_tokenize failed"); return nullptr; }
    ids.resize(n);

    jintArray out = env->NewIntArray(n);
    env->SetIntArrayRegion(out, 0, n, reinterpret_cast<jint*>(ids.data()));
    return out;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_zhangke_llama_Llama_nativeDetokenize(
        JNIEnv* env, jclass, jintArray jtokens) {
    if (!ensure_model(env)) return nullptr;
    jsize n = env->GetArrayLength(jtokens);
    std::vector<llama_token> toks(n);
    env->GetIntArrayRegion(jtokens, 0, n, reinterpret_cast<jint*>(toks.data()));
    std::string text = detok(g_model, toks);
    return env->NewStringUTF(text.c_str());
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_zhangke_llama_Llama_nativeEmbeddings(
        JNIEnv* env, jclass, jstring jtext) {
    if (!ensure_ctx(env)) return nullptr;
    const char* ctext = env->GetStringUTFChars(jtext, nullptr);

    // tokenize
    int cap = (int)strlen(ctext) + 8;
    std::vector<llama_token> inp(cap);
    int n_inp = llama_tokenize(g_model, ctext, inp.data(), cap, /*add_bos=*/true);
    env->ReleaseStringUTFChars(jtext, ctext);
    if (n_inp < 0) { jthrow(env, "llama_tokenize failed"); return nullptr; }
    inp.resize(n_inp);

    // eval prompt（分批）
    int n_past = 0;
    const int n_batch = 32;
    for (int i = 0; i < n_inp; i += n_batch) {
        int n_eval = std::min(n_batch, n_inp - i);
        if (llama_eval(g_ctx, inp.data() + i, n_eval, n_past, g_n_threads)) {
            jthrow(env, "llama_eval failed"); return nullptr;
        }
        n_past += n_eval;
    }

    // 取向量（需要在 context params 开启 embeddings）
    const float* emb = llama_get_embeddings(g_ctx);
    if (!emb) { jthrow(env, "embeddings not available"); return nullptr; }
    int dim = llama_n_embd(g_model);

    jfloatArray out = env->NewFloatArray(dim);
    env->SetFloatArrayRegion(out, 0, dim, emb);
    return out;
}

// ================== 采样辅助 ==================
static llama_token sample_next_token(
        llama_context* ctx,
        float temperature, float top_p, int top_k,
        float repeat_penalty, float freq_penalty, float pres_penalty,
        const std::deque<llama_token>& last_tokens) {

    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const float* logits = llama_get_logits(ctx);

    std::vector<llama_token_data> cands;
    cands.reserve(n_vocab);
    for (int id = 0; id < n_vocab; ++id) cands.push_back({ id, logits[id], 0.0f });
    llama_token_data_array arr = { cands.data(), cands.size(), false };

    if (repeat_penalty != 1.0f || freq_penalty != 0.0f || pres_penalty != 0.0f) {
        llama_sample_repetition_penalty(ctx, &arr,
                                        last_tokens.data(), (int)last_tokens.size(),
                                        repeat_penalty, freq_penalty, pres_penalty);
    }
    if (top_k > 0)      llama_sample_top_k(ctx, &arr, top_k, 1);
    if (top_p < 1.0f)   llama_sample_top_p(ctx, &arr, top_p, 1);
    if (temperature > 0 && temperature != 1.0f)
        llama_sample_temperature(ctx, &arr, temperature);

    return llama_sample_token(ctx, &arr); // 温度=0 时等价于 argmax
}

// ================== 一次性生成 ==================
extern "C" JNIEXPORT jstring JNICALL
Java_com_zhangke_llama_Llama_nativeGenerate(
        JNIEnv* env, jclass,
        jstring jprompt,
        jint maxTokens, jfloat temperature, jfloat topP, jint topK,
        jfloat repeatPenalty, jfloat freqPenalty, jfloat presPenalty,
        jobjectArray jstops) {

    if (!ensure_ctx(env)) return nullptr;

    // stop 词
    std::vector<std::string> stops;
    if (jstops) {
        jsize nstop = env->GetArrayLength(jstops);
        for (jsize i = 0; i < nstop; ++i) {
            auto jstr = (jstring) env->GetObjectArrayElement(jstops, i);
            const char* cs = env->GetStringUTFChars(jstr, nullptr);
            stops.emplace_back(cs ? cs : "");
            env->ReleaseStringUTFChars(jstr, cs);
            env->DeleteLocalRef(jstr);
        }
    }

    // tokenize prompt
    const char* cprompt = env->GetStringUTFChars(jprompt, nullptr);
    int cap = (int)strlen(cprompt) + 8;
    std::vector<llama_token> inp(cap);
    int n_inp = llama_tokenize(g_model, cprompt, inp.data(), cap, /*add_bos=*/true);
    env->ReleaseStringUTFChars(jprompt, cprompt);
    if (n_inp < 0) { jthrow(env, "llama_tokenize failed"); return nullptr; }
    inp.resize(n_inp);

    // eval prompt
    int n_past = 0;
    const int n_batch = 32;
    for (int i = 0; i < n_inp; i += n_batch) {
        int n_eval = std::min(n_batch, n_inp - i);
        if (llama_eval(g_ctx, inp.data() + i, n_eval, n_past, g_n_threads)) {
            jthrow(env, "llama_eval failed"); return nullptr;
        }
        n_past += n_eval;
    }

    std::deque<llama_token> last_tokens;
    last_tokens.insert(last_tokens.end(), inp.begin(), inp.end());
    const int last_n = std::min(64, g_n_ctx);

    std::string out;
    out.reserve(maxTokens * 4);
    g_cancel.store(false);

    for (int i = 0; i < maxTokens; ++i) {
        llama_token tok = sample_next_token(
                g_ctx, temperature, topP, topK,
                repeatPenalty, freqPenalty, presPenalty,
                last_tokens
        );
        if (tok == llama_token_eos(g_model)) break;

        std::string piece = token_to_piece(g_model, tok);
        out += piece;

        // stop 词后缀匹配
        bool stop_hit = false;
        for (auto& s : stops) {
            if (!s.empty() && ends_with(out, s)) {
                out.resize(out.size() - s.size());
                stop_hit = true; break;
            }
        }
        if (stop_hit) break;

        // 继续 eval 该 token（自回归）
        if (llama_eval(g_ctx, &tok, 1, n_past, g_n_threads)) {
            jthrow(env, "llama_eval failed during generation"); return nullptr;
        }
        n_past += 1;

        last_tokens.push_back(tok);
        if ((int)last_tokens.size() > last_n) last_tokens.pop_front();

        if (g_cancel.load()) break;
    }

    return env->NewStringUTF(out.c_str());
}

// ================== 流式生成 ==================
extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeGenerateStreaming(
        JNIEnv* env, jclass,
        jstring jprompt,
        jint maxTokens, jfloat temperature, jfloat topP, jint topK,
        jfloat repeatPenalty, jfloat freqPenalty, jfloat presPenalty,
        jobjectArray jstops, jobject jcallback) {

    if (!ensure_ctx(env)) return;
    if (!g_CallbackCls || !g_OnTokenMethod || !g_ShouldStop) {
        jthrow(env, "Callback methods not resolved"); return;
    }
    if (!jcallback) { jthrow(env, "Callback is null"); return; }

    jobject cb = env->NewGlobalRef(jcallback);

    // stop 词
    std::vector<std::string> stops;
    if (jstops) {
        jsize nstop = env->GetArrayLength(jstops);
        for (jsize i = 0; i < nstop; ++i) {
            auto jstr = (jstring) env->GetObjectArrayElement(jstops, i);
            const char* cs = env->GetStringUTFChars(jstr, nullptr);
            stops.emplace_back(cs ? cs : "");
            env->ReleaseStringUTFChars(jstr, cs);
            env->DeleteLocalRef(jstr);
        }
    }

    // tokenize prompt
    const char* cprompt = env->GetStringUTFChars(jprompt, nullptr);
    int cap = (int)strlen(cprompt) + 8;
    std::vector<llama_token> inp(cap);
    int n_inp = llama_tokenize(g_model, cprompt, inp.data(), cap, /*add_bos=*/true);
    env->ReleaseStringUTFChars(jprompt, cprompt);
    if (n_inp < 0) { jthrow(env, "llama_tokenize failed"); env->DeleteGlobalRef(cb); return; }
    inp.resize(n_inp);

    // eval prompt
    int n_past = 0;
    const int n_batch = 32;
    for (int i = 0; i < n_inp; i += n_batch) {
        int n_eval = std::min(n_batch, n_inp - i);
        if (llama_eval(g_ctx, inp.data() + i, n_eval, n_past, g_n_threads)) {
            jthrow(env, "llama_eval failed"); env->DeleteGlobalRef(cb); return;
        }
        n_past += n_eval;
    }

    std::deque<llama_token> last_tokens;
    last_tokens.insert(last_tokens.end(), inp.begin(), inp.end());
    const int last_n = std::min(64, g_n_ctx);

    std::string assembled;
    assembled.reserve(maxTokens * 4);
    g_cancel.store(false);

    for (int i = 0; i < maxTokens; ++i) {
        llama_token tok = sample_next_token(
                g_ctx, temperature, topP, topK,
                repeatPenalty, freqPenalty, presPenalty,
                last_tokens
        );
        if (tok == llama_token_eos(g_model)) break;

        std::string piece = token_to_piece(g_model, tok);
        assembled += piece;

        // stop 词检查（在回调前去掉）
        bool stop_hit = false;
        for (auto& s : stops) {
            if (!s.empty() && ends_with(assembled, s)) {
                assembled.resize(assembled.size() - s.size());
                stop_hit = true; break;
            }
        }

        // 回调给 Kotlin
        jstring jpiece = env->NewStringUTF(piece.c_str());
        env->CallVoidMethod(cb, g_OnTokenMethod, jpiece, (jint)tok);
        env->DeleteLocalRef(jpiece);

        if (env->ExceptionCheck()) { // 上层抛异常则中断
            env->ExceptionClear();
            break;
        }
        if (env->CallBooleanMethod(cb, g_ShouldStop)) break;
        if (stop_hit) break;

        // 继续 eval
        if (llama_eval(g_ctx, &tok, 1, n_past, g_n_threads)) {
            jthrow(env, "llama_eval failed during streaming");
            env->DeleteGlobalRef(cb); return;
        }
        n_past += 1;

        last_tokens.push_back(tok);
        if ((int)last_tokens.size() > last_n) last_tokens.pop_front();

        if (g_cancel.load()) break;
    }

    env->DeleteGlobalRef(cb);
}

// =========（可选）取消与线程设置，如需要请在 Kotlin 侧也声明 external =========
extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeCancel(JNIEnv*, jclass) {
    g_cancel.store(true);
}
extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeSetThreads(JNIEnv*, jclass, jint n_threads) {
    g_n_threads = std::max(1, (int)n_threads);
}