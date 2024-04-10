package com.dicoding.asclepius.view

import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.dicoding.asclepius.databinding.ActivityResultBinding
import com.dicoding.asclepius.helper.ImageClassifierHelper
import org.tensorflow.lite.task.vision.classifier.Classifications

class ResultActivity : AppCompatActivity() {
    private lateinit var binding: ActivityResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val imageUriString = intent.getStringExtra(EXTRA_IMAGE_URI)
        val imageUri = Uri.parse(imageUriString)
        imageUri?.let {
            Log.d(TAG, "ShowImage: $it")
            binding.resultImage.setImageURI(it)

            val imageClassifierHelper = ImageClassifierHelper(
                context = this,
                classifierListener = object : ImageClassifierHelper.ClassifierListener {
                    override fun onError(error: String) {
                        Log.d(TAG, "Error: $error")
                    }

                    override fun onResults(results: List<Classifications>?, inferenceTime: Long) {
                        results?.let {
                            val topResult = it[0]
                            val label = topResult.categories[0].label
                            val score = topResult.categories[0].score

                            binding.resultText.text = "$label ${(score * 100).toInt()}%"
                        }
                    }
                }
            )
            imageClassifierHelper.classifyStaticImage(imageUri)
        }

        // Add back button functionality
        binding.backButton.setOnClickListener {
            onBackPressed()
        }
    }

    companion object {
        const val EXTRA_IMAGE_URI = "extra_image_uri"
        const val TAG = "ImagePicker"
    }
}
