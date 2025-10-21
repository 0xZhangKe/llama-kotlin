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

    data class GenOptions(
        val maxTokens: Int = 256,
        val temperature: Float = 0.8f,
        val topP: Float = 0.95f,
        val topK: Int = 40,
        val repeatPenalty: Float = 1.1f,
        val frequencyPenalty: Float = 0.0f,
        val presencePenalty: Float = 0.0f,
        val stop: Array<String> = emptyArray()
    )

    interface TokenCallback {
        fun onToken(text: String, tokenId: Int) {}
        fun shouldStop(): Boolean = false
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
    fun tokenize(text: String, addBos: Boolean = true): IntArray =
        nativeTokenize(text, addBos)
    fun detokenize(tokens: IntArray): String =
        nativeDetokenize(tokens)
    fun embeddings(text: String): FloatArray =
        nativeEmbeddings(text)

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
        callback: TokenCallback
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
    @Keep @JvmStatic external fun nativeLoadModel(
        modelPath: String,
        nCtx: Int, nGpuLayers: Int, nThreads: Int,
        seed: Int, useMMap: Boolean, useMLock: Boolean
    )

    @Keep @JvmStatic external fun nativeFreeModel()

    @Keep @JvmStatic external fun nativeReset()
    @Keep @JvmStatic external fun nativeVocabSize(): Int
    @Keep @JvmStatic external fun nativeTokenize(text: String, addBos: Boolean): IntArray
    @Keep @JvmStatic external fun nativeDetokenize(tokens: IntArray): String
    @Keep @JvmStatic external fun nativeEmbeddings(text: String): FloatArray

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
        callback: TokenCallback
    )
}
