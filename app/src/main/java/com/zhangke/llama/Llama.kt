package com.zhangke.llama

import androidx.annotation.Keep

object Llama {

    init {
        System.loadLibrary("llama_jni")
    }

    data class InitParams(
        val nCtx: Int = 4096,
        val nGpuLayers: Int = 0,
        val nThreads: Int = Runtime.getRuntime().availableProcessors(),
        val seed: Int = -1,
        val useMMap: Boolean = true,
        val useMLock: Boolean = false
    )

    class GenOptions(
        val maxTokens: Int = 256,
        val temperature: Float = 0.8f,
        val topP: Float = 0.95f,
        val topK: Int = 40,
        val repeatPenalty: Float = 1.1f,
        val frequencyPenalty: Float = 0.0f,
        val presencePenalty: Float = 0.0f,
        val stop: Array<String> = emptyArray()
    )

    interface GenerateCallback {
        fun onDelta(text: String)
        fun onDone()
    }

    // ---- Kotlin API ----
    fun load(modelPath: String, params: InitParams = InitParams()) {
        nativeLoadModel(
            modelPath,
            params.nCtx, params.nGpuLayers, params.nThreads,
            params.seed, params.useMMap, params.useMLock
        )
    }

    fun unload() = nativeFreeModel()
    fun reset() = nativeReset()

    fun vocabSize(): Int = nativeVocabSize()

    fun tokenize(text: String, addBos: Boolean = true): IntArray = nativeTokenize(text, addBos)

    fun detokenize(tokens: IntArray): String = nativeDetokenize(tokens)

    fun generate(prompt: String, options: GenOptions = GenOptions()): String =
        nativeGenerate(
            prompt,
            options.maxTokens, options.temperature, options.topP, options.topK,
            options.repeatPenalty, options.frequencyPenalty, options.presencePenalty,
            options.stop
        )

    fun generateStreaming(
        prompt: String,
        options: GenOptions = GenOptions(),
        callback: GenerateCallback
    ) {
        nativeGenerateStreaming(
            prompt,
            options.maxTokens, options.temperature, options.topP, options.topK,
            options.repeatPenalty, options.frequencyPenalty, options.presencePenalty,
            options.stop,
            callback
        )
    }

    // ---- JNI ----
    /**
     * @param nCtx 上下文长度（context size）控制模型一次能记住多少 token；常用值 2048~8192。
     * @param nGpuLayers GPU 层数（仅在支持 GPU 的 llama 变体中有效）决定把前多少层模型加载到 GPU 上。手机上通常填 0（全部 CPU）。
     * @param nThreads 推理时使用的 CPU 线程数 llama_eval 会用它控制并行度。填 4 就代表使用 4 个线程。
     * @param useMMap 是否使用内存映射（memory-mapped file）加载模型 true 可以显著减少启动时内存占用（因为不需要完整读入文件），但文件必须位于支持 mmap 的文件系统上。
     * @param useMLock 是否锁定模型到物理内存（防止被换出）一般在服务器或桌面上启用。手机上建议关掉。
     */
    @Keep @JvmStatic external fun nativeLoadModel(
        modelPath: String,
        nCtx: Int, nGpuLayers: Int, nThreads: Int,
        useMMap: Boolean, useMLock: Boolean
    )

    @Keep @JvmStatic external fun nativeFreeModel()

    @Keep @JvmStatic external fun nativeReset()
    @Keep @JvmStatic external fun nativeVocabSize(): Int
    @Keep @JvmStatic external fun nativeTokenize(text: String, addBos: Boolean): IntArray
    @Keep @JvmStatic external fun nativeDetokenize(tokens: IntArray): String

    @Keep @JvmStatic external fun nativeGenerate(
        prompt: String,
        maxTokens: Int, temperature: Float, topP: Float, topK: Int,
        repeatPenalty: Float, frequencyPenalty: Float, presencePenalty: Float,
        stop: Array<String>
    ): String

    @Keep @JvmStatic external fun nativeGenerateStreaming(
        prompt: String,
        maxTokens: Int, temperature: Float, topP: Float, topK: Int,
        repeatPenalty: Float, frequencyPenalty: Float, presencePenalty: Float,
        stop: Array<String>,
        callback: GenerateCallback
    )
}
