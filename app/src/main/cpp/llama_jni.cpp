// src/main/cpp/llama_jni.cpp
#include <jni.h>
#include <string>
#include <vector>
#include <deque>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <limits>
#include <cmath>

#include "llama.h"   // 确保 CMake 已把 llama.cpp 的头文件目录加入 include 路径

// ================== 全局状态（与 Kotlin 的 object 匹配） ==================
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static int g_n_threads = 4;
static int g_n_ctx = 4096;
static std::atomic<bool> g_cancel{false};

// 回调缓存
static jclass g_CallbackCls = nullptr;
static jmethodID g_OnTokenMethod = nullptr; // void onToken(String,int)
static jmethodID g_ShouldStop = nullptr; // boolean shouldStop()

// --------- 工具 ----------
static void jthrow(JNIEnv *env, const char *msg) {
    jclass ex = env->FindClass("java/lang/RuntimeException");
    if (ex) env->ThrowNew(ex, msg);
}

static bool ensure_model(JNIEnv *env) {
    if (!g_model) {
        jthrow(env, "Model is null. Call Llama.load() first.");
        return false;
    }
    return true;
}

static bool ensure_ctx(JNIEnv *env) {
    if (!g_ctx) {
        jthrow(env, "Context is null. Call Llama.load() first.");
        return false;
    }
    return true;
}

static bool ends_with(const std::string &s, const std::string &suf) {
    return s.size() >= suf.size() && std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}

static std::string token_to_piece(const llama_model* model, llama_token tok) {
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int need = llama_detokenize(
            vocab,
            &tok, 1,
            /*text=*/nullptr, /*text_len_max=*/0,
            /*remove_special=*/false,
            /*unparse_special=*/false
    );
    if (need <= 0) return {};

    std::string out;
    out.resize(need);

    const int wrote = llama_token_to_piece(
            vocab,
            tok,
            out.data(),
            (int)out.size(),
            /*lstrip=*/0,
            /*special=*/true
    );
    if (wrote < 0) return {};
    if (wrote < (int)out.size()) out.resize(wrote);
    return out;
}

// 多 token 简易 detok
static std::string detok(llama_model *model, const std::vector<llama_token> &toks) {
    std::string s;
    s.reserve(toks.size() * 4);
    for (auto t: toks) s += token_to_piece(model, t);
    return s;
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
    JNIEnv *env = nullptr;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    jclass cb = env->FindClass("com/zhangke/llama/Llama$TokenCallback");
    if (cb) {
        g_CallbackCls = (jclass) env->NewGlobalRef(cb);
        g_OnTokenMethod = env->GetMethodID(g_CallbackCls, "onToken", "(Ljava/lang/String;I)V");
        g_ShouldStop = env->GetMethodID(g_CallbackCls, "shouldStop", "()Z");
    }

    llama_backend_init(); // llama 全局初始化
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
    JNIEnv *env = nullptr;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) == JNI_OK) {
        if (g_CallbackCls) {
            env->DeleteGlobalRef(g_CallbackCls);
            g_CallbackCls = nullptr;
        }
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
}

// ================== Llama.load / unload ==================
extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeLoadModel(
        JNIEnv *env, jclass,
        jstring jpath, jint n_ctx, jint n_gpu_layers, jint n_threads,
        jboolean use_mmap, jboolean use_mlock) {

    // 允许重复 load：先清理旧的
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    const char *cpath = env->GetStringUTFChars(jpath, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_mmap = (bool) use_mmap;
    mparams.use_mlock = (bool) use_mlock;

    g_model = llama_model_load_from_file(cpath, mparams);
    env->ReleaseStringUTFChars(jpath, cpath);
    if (!g_model) {
        jthrow(env, "llama_load_model_from_file failed");
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.embeddings = true;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        jthrow(env, "llama_init_from_model failed");
        return;
    }

    g_n_threads = std::max(1, (int) n_threads);
    g_n_ctx = n_ctx;
    g_cancel.store(false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeFreeModel(JNIEnv *, jclass) {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_cancel.store(false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeReset(JNIEnv *env, jclass) {
    if (!ensure_ctx(env)) return;
    llama_memory_t mem = llama_get_memory(g_ctx);
    llama_memory_clear(mem, true);
    g_cancel.store(false);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_zhangke_llama_Llama_nativeVocabSize(JNIEnv *env, jclass) {
    if (!ensure_model(env)) return 0;
    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);
    return (jint) llama_vocab_n_tokens(vocab);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_zhangke_llama_Llama_nativeTokenize(JNIEnv *env, jclass, jstring jtext) {
    if (!ensure_model(env)) return nullptr;

    const char *text = env->GetStringUTFChars(jtext, nullptr);
    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    std::vector<llama_token> tokens(1024);
    int n = llama_tokenize(vocab, text, strlen(text), tokens.data(), tokens.size(), true, false);
    env->ReleaseStringUTFChars(jtext, text);

    jintArray result = env->NewIntArray(n);
    env->SetIntArrayRegion(result, 0, n, reinterpret_cast<jint *>(tokens.data()));
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_zhangke_llama_Llama_nativeDetokenize(
        JNIEnv *env, jclass, jintArray jtokens) {
    if (!ensure_model(env)) return nullptr;
    jsize n = env->GetArrayLength(jtokens);
    std::vector<llama_token> toks(n);
    env->GetIntArrayRegion(jtokens, 0, n, reinterpret_cast<jint *>(toks.data()));
    std::string text = detok(g_model, toks);
    return env->NewStringUTF(text.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_zhangke_llama_Llama_nativeGenerate(
        JNIEnv *env, jclass,
        jstring jprompt,
        jint maxTokens, jfloat /*temperature*/, jfloat /*topP*/, jint /*topK*/,
        jfloat /*repeatPenalty*/, jfloat /*freqPenalty*/, jfloat /*presPenalty*/,
        jobjectArray /*jstops*/) {

    if (!ensure_ctx(env)) return env->NewStringUTF("");

    llama_context *ctx = g_ctx;
    const llama_model *model = g_model;
    const llama_vocab *vocab = llama_model_get_vocab(model);

    const int eos_id = llama_token_eos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char *cprompt = env->GetStringUTFChars(jprompt, nullptr);
    std::vector<llama_token> prompt_tokens(1024);
    int nt = llama_tokenize(
            vocab,
            cprompt, (int) strlen(cprompt),
            prompt_tokens.data(), (int) prompt_tokens.size(),
            /*add_special=*/true, /*parse_special=*/false
    );
    env->ReleaseStringUTFChars(jprompt, cprompt);
    if (nt <= 0) return env->NewStringUTF("");

    llama_batch batch = llama_batch_init(nt + maxTokens, 0, 1);

    llama_pos pos = 0;
    for (int i = 0; i < nt; ++i) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = pos++;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == nt - 1) ? 1 : 0;
    }
    batch.n_tokens = nt;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return env->NewStringUTF("");
    }

    std::string out;
    out.reserve(1024);

    for (int t = 0; t < (int) maxTokens; ++t) {
        const float *logits = nullptr;
        logits = llama_get_logits(ctx);
        if (!logits) break;

        int next_id = int(std::max_element(logits, logits + n_vocab) - logits);
        if (next_id == eos_id) break;

        // detokenize 追加输出
        char piece[256];
        llama_token_to_piece(vocab, (llama_token) next_id, piece, (int) sizeof(piece), /*lstrip=*/
                             0, /*special=*/false);
        out += piece;

        llama_batch_free(batch);
        batch = llama_batch_init(1, 0, 1);

        batch.token[0] = (llama_token) next_id;
        batch.pos[0] = pos++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        if (llama_decode(ctx, batch) != 0) break;
    }
    llama_batch_free(batch);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_zhangke_llama_Llama_nativeGenerateStreaming(
        JNIEnv* env, jclass,
        jstring jprompt,
        jint maxTokens, jfloat /*temperature*/, jfloat /*topP*/, jint /*topK*/,
        jfloat /*repeatPenalty*/, jfloat /*freqPenalty*/, jfloat /*presPenalty*/,
        jobjectArray /*jstops*/,
        jobject jcallback) {

    if (!ensure_ctx(env)) return;
    if (jcallback == nullptr) return;

    jclass cbCls = env->GetObjectClass(jcallback);
    jmethodID midOnDelta = env->GetMethodID(cbCls, "onDelta", "(Ljava/lang/String;)V");
    jmethodID midOnDone  = env->GetMethodID(cbCls, "onDone",  "()V");
    if (!midOnDelta || !midOnDone) return;

    llama_context* ctx   = g_ctx;
    const llama_model* model = g_model;
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int eos_id  = llama_vocab_eos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char* cprompt = env->GetStringUTFChars(jprompt, nullptr);
    std::vector<llama_token> prompt_tokens(1024);
    int nt = llama_tokenize(
            vocab,
            cprompt, (int)strlen(cprompt),
            prompt_tokens.data(), (int)prompt_tokens.size(),
            /*add_special=*/true, /*parse_special=*/false
    );
    env->ReleaseStringUTFChars(jprompt, cprompt);
    if (nt <= 0) {
        env->CallVoidMethod(jcallback, midOnDone);
        return;
    }

    llama_pos pos = 0;
    llama_batch batch = llama_batch_init(/*n_tokens_max=*/nt + std::max(1, (int)maxTokens),
            /*embd=*/0,
            /*n_seq_max=*/1);

    for (int i = 0; i < nt; ++i) {
        batch.token[i]     = prompt_tokens[i];
        batch.pos[i]       = pos++;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == nt - 1) ? 1 : 0;
    }
    batch.n_tokens = nt;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        env->CallVoidMethod(jcallback, midOnDone);
        return;
    }

    for (int t = 0; t < (int)maxTokens; ++t) {
        const float* logits = llama_get_logits(ctx);
        if (!logits) break;

        int next_id = int(std::max_element(logits, logits + n_vocab) - logits);
        if (next_id == eos_id) break;

        char piece[512];
        llama_token_to_piece(vocab, (llama_token)next_id, piece, (int)sizeof(piece),
                /*lstrip=*/0, /*special=*/false);
        jstring jpiece = env->NewStringUTF(piece);
        env->CallVoidMethod(jcallback, midOnDelta, jpiece);
        env->DeleteLocalRef(jpiece);

        batch.n_tokens     = 0;
        batch.token[0]     = (llama_token)next_id;
        batch.pos[0]       = pos++;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;

        if (llama_decode(ctx, batch) != 0) break;
    }

    llama_batch_free(batch);
    env->CallVoidMethod(jcallback, midOnDone);
}
