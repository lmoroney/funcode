package com.lmoroney.modelmakertextclassifier

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.lmoroney.modelmakertextclassifier.helpers.Result
import com.lmoroney.modelmakertextclassifier.helpers.TextClassificationClient

class MainActivity : AppCompatActivity() {
    lateinit var txtInput:EditText
    lateinit var btnClassify:Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val client = TextClassificationClient(applicationContext)
        client.load()
        txtInput = findViewById(R.id.txtInput)
        btnClassify = findViewById(R.id.btnClassify)
        btnClassify.setOnClickListener {
            var toClassify:String = txtInput.text.toString()
            var results:List<Result> = client.classify(toClassify)
            showResult(toClassify, results)
        }

    }

    /** Show classification result on the screen.  */
    private fun showResult(inputText: String, results: List<Result>) {
        // Run on UI thread as we'll updating our app UI
        runOnUiThread {
            var textToShow = "Input: $inputText\nOutput:\n"
            for (i in results.indices) {
                val result = results[i]
                textToShow += java.lang.String.format(
                    "    %s: %s\n",
                    result.title,
                    result.confidence
                )
            }
            textToShow += "---------\n"

            Toast.makeText(this, textToShow, Toast.LENGTH_LONG).show()
        }
    }
}