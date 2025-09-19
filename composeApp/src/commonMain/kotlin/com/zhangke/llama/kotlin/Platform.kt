package com.zhangke.llama.kotlin

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform