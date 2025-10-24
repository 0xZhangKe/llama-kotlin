package com.zhangke.llama.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.zhangke.llama.Llama
import com.zhangke.llama.Llama.GenerateCallback
import com.zhangke.llama.app.ui.theme.LlamakotlinTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val modelFile =
            File(getExternalFilesDir(null), "models/Mistral-7B-Instruct-v0.3-IQ1_M.gguf")

        setContent {
            LlamakotlinTheme {
                var loadResult by rememberSaveable { mutableStateOf("") }
                var chatLog by rememberSaveable { mutableStateOf("") }
                MainPage(
                    modelFile = modelFile,
                    loadResult = loadResult,
                    chatLog = chatLog,
                    onLoadModel = {
                        runCatching {
                            Llama.load(modelFile.absolutePath)
                        }.onSuccess {
                            loadResult = "load success"
                        }.onFailure {
                            loadResult = "load failed"
                        }
                    },
                    onSendPrompt = { prompt ->
                        chatLog = "start generate...\n"
                        lifecycleScope.launch(Dispatchers.IO) {
                            runCatching {
                                val start = System.currentTimeMillis()
                                Llama.generateStreaming(
                                    prompt = prompt,
                                    callback = object : GenerateCallback {

                                        override fun onDelta(text: String) {
                                            chatLog += text
                                        }

                                        override fun onDone() {
                                            chatLog += "\n===done===\n"
                                            chatLog += "Time taken: ${(System.currentTimeMillis() - start) / 1000} seconds\n"
                                        }
                                    },
                                )
                            }.onSuccess {
                                chatLog += "generate finished."
                            }.onFailure {
                                chatLog += "generate failed: ${it.message}."
                            }
                        }
                    },
                    onReleaseModel = {
                        runCatching {
                            Llama.unload()
                        }.onSuccess {
                            loadResult = "release success"
                        }.onFailure {
                            loadResult = "release failed"
                        }
                    },
                )
            }
        }
    }
}


@Composable
private fun MainPage(
    modelFile: File,
    loadResult: String,
    chatLog: String,
    onLoadModel: () -> Unit,
    onSendPrompt: (String) -> Unit,
    onReleaseModel: () -> Unit,
) {
    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Box(
            modifier = Modifier
                .padding(innerPadding)
                .padding(horizontal = 16.dp)
                .fillMaxSize()
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text = "Model: ${modelFile.name}",
                    style = MaterialTheme.typography.labelSmall,
                )
                Spacer(modifier = Modifier.height(16.dp))
                if (modelFile.exists()) {
                    Text(
                        text = "Model file size: ${(modelFile.length() / 1024 / 1024)} MB",
                        style = MaterialTheme.typography.labelSmall,
                    )
                } else {
                    Text(
                        text = "Model file does not exist.",
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Model file load result: $loadResult",
                    style = MaterialTheme.typography.labelSmall,
                )

                Box(
                    modifier = Modifier
                        .padding(top = 16.dp, end = 16.dp)
                        .weight(1F),
                ) {
                    Text(
                        text = chatLog,
                        style = MaterialTheme.typography.labelSmall,
                    )
                }

                Button(
                    onClick = onLoadModel,
                ) {
                    Text("Load Model")
                }

                Button(
                    onClick = { onSendPrompt("Hello, Who are you") },
                ) {
                    Text("Send Prompt")
                }

                Button(
                    onClick = onReleaseModel,
                ) {
                    Text("Release Model")
                }
            }
        }
    }
}
