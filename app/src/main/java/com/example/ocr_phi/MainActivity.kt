package com.example.ocr_phi

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import kotlin.time.TimeSource
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils.bitmapToMat
import org.opencv.android.Utils.matToBitmap
import org.opencv.core.*
import org.opencv.dnn.Dnn.NMSBoxesRotated
import org.opencv.imgproc.Imgproc.boxPoints
import org.opencv.imgproc.Imgproc.getPerspectiveTransform
import org.opencv.imgproc.Imgproc.warpPerspective
import org.opencv.utils.Converters.vector_RotatedRect_to_Mat
import org.opencv.utils.Converters.vector_float_to_Mat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

class MainActivity : AppCompatActivity() {
    private lateinit var llmInference: LlmInference
    private lateinit var inputEditText: EditText
    private lateinit var sendButton: Button
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var chatAdapter: ChatAdapter
    private val chatMessages = mutableListOf<ChatMessage>()

    private lateinit var trOcrDetector: Interpreter
    private lateinit var trOcrRecognizer: Interpreter
    private lateinit var cameraButton: Button
    private lateinit var galleryButton: Button
    private lateinit var indicesMat: MatOfInt
    private val alphabets = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@#/%+=><?-!"
    private val detectionImageHeight = 320
    private val detectionImageWidth = 320
    private val detectionImageMeans = floatArrayOf(103.94f, 116.78f, 123.68f)
    private val detectionImageStds = floatArrayOf(1f, 1f, 1f)
    private val detectionOutputNumRows = 80
    private val detectionOutputNumCols = 80
    private val detectionConfidenceThreshold = 0.5f
    private val detectionNMSThreshold = 0.4f
    private val recognitionImageHeight = 31
    private val recognitionImageWidth = 200
    private val recognitionImageMean = 0f
    private val recognitionImageStd = 255f
    private val recognitionModelOutputSize = 48
    private var recognitionResult = ByteBuffer.allocateDirect(recognitionModelOutputSize * 8)
    private var ocrResults: HashMap<String, Int> = HashMap()

    private val CAMERA_PERMISSION_CODE = 100
    private val CAMERA_REQUEST_CODE = 101
    private val GALLERY_REQUEST_CODE = 102

    private var conversationHistory = mutableListOf<String>()
    private var currentOcrResults = mutableListOf<String>()

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        OpenCVLoader.initDebug()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputEditText = findViewById(R.id.inputEditText)
        sendButton = findViewById(R.id.sendButton)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)
        cameraButton = findViewById(R.id.cameraButton)
        galleryButton = findViewById(R.id.galleryButton)

        chatAdapter = ChatAdapter(chatMessages)
        chatRecyclerView.adapter = chatAdapter
        chatRecyclerView.layoutManager = LinearLayoutManager(this)

        recognitionResult.order(ByteOrder.nativeOrder())

        val options = LlmInferenceOptions.builder()
            .setModelPath("/data/local/tmp/phi2_cpu.bin")
            .setMaxTokens(512)
            .setTopK(30)
            .setTemperature(0.4F)
            .setRandomSeed(101)
            .build()

        llmInference = LlmInference.createFromOptions(this, options)

        try {
            trOcrDetector = Interpreter(loadModelFile("1.tflite"))
            trOcrRecognizer = Interpreter(loadModelFile("2.tflite"))
        } catch (e: Exception) {
            e.printStackTrace()
            chatMessages.add(ChatMessage("Error loading model: ${e.message}", false))
            chatAdapter.notifyItemInserted(chatMessages.size - 1)
        }

        sendButton.setOnClickListener { performLLMInference() }
        cameraButton.setOnClickListener { openCamera() }
        galleryButton.setOnClickListener { openGallery() }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun openCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE)
        } else {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, CAMERA_REQUEST_CODE)
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, GALLERY_REQUEST_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                CAMERA_REQUEST_CODE -> {
                    val imageBitmap = data?.extras?.get("data") as Bitmap?
                    if (imageBitmap != null) {
                        performOCR(imageBitmap)
                    }
                }
                GALLERY_REQUEST_CODE -> {
                    val selectedImage: Uri? = data?.data
                    if (selectedImage != null) {
                        val imageBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImage)
                        performOCR(imageBitmap)
                    }
                }
            }
        }
    }

    private fun performOCR(bitmapImage: Bitmap) {
        val timeSource = TimeSource.Monotonic

        val markD1 = timeSource.markNow()
        val detectionResults = detectTexts(bitmapImage)
        val markD2 = timeSource.markNow()
        val markD = markD2 - markD1

        Log.d("Time", "Detection Time: $markD")

        val markR1 = timeSource.markNow()
        currentOcrResults.clear() // Clear previous results
        ocrResults.clear() // Clear previous results
        val resultBitmap = recognizeTexts(bitmapImage, detectionResults.first, detectionResults.second)
        val markR2 = timeSource.markNow()
        val markR = markR2 - markR1

        Log.d("Time", "Recognition Time: $markR")

        chatMessages.add(ChatMessage(resultBitmap, false, true))
        chatAdapter.notifyItemInserted(chatMessages.size - 1)
        chatRecyclerView.scrollToPosition(chatMessages.size - 1)

        val ocrText = currentOcrResults.joinToString("\n")
        performLLMInference(ocrText)
    }

    private fun detectTexts(data: Bitmap): Pair<MatOfRotatedRect, MatOfInt> {
        val detectorTensorImage = preprocessImage(data, detectionImageWidth, detectionImageHeight, detectionImageMeans, detectionImageStds)
        val detectionInputs = arrayOf(detectorTensorImage.buffer.rewind())
        val detectionOutputs: HashMap<Int, Any> = HashMap()

        val ratioHeight = data.height.toFloat() / detectionImageHeight
        val ratioWidth = data.width.toFloat() / detectionImageWidth

        indicesMat = MatOfInt()

        val detectionScores = Array(1) { Array(detectionOutputNumRows) { Array(detectionOutputNumCols) { FloatArray(1) } } }
        val detectionGeometries = Array(1) { Array(detectionOutputNumRows) { Array(detectionOutputNumCols) { FloatArray(5) } } }
        detectionOutputs[0] = detectionScores
        detectionOutputs[1] = detectionGeometries

        trOcrDetector.runForMultipleInputsOutputs(detectionInputs, detectionOutputs)

        val transposedDetectionScores = Array(1) { Array(1) { Array(detectionOutputNumRows) { FloatArray(detectionOutputNumCols) } } }
        val transposedDetectionGeometries = Array(1) { Array(5) { Array(detectionOutputNumRows) { FloatArray(detectionOutputNumCols) } } }

        for (i in 0 until transposedDetectionScores[0][0].size) {
            for (j in 0 until transposedDetectionScores[0][0][0].size) {
                for (k in 0 until 1) {
                    transposedDetectionScores[0][k][i][j] = detectionScores[0][i][j][k]
                }
                for (k in 0 until 5) {
                    transposedDetectionGeometries[0][k][i][j] = detectionGeometries[0][i][j][k]
                }
            }
        }

        val detRotatedRects = ArrayList<RotatedRect>()
        val detConfidences = ArrayList<Float>()

        for (y in 0 until transposedDetectionScores[0][0].size) {
            val detectionScoreData = transposedDetectionScores[0][0][y]
            val detectionGeometryX0Data = transposedDetectionGeometries[0][0][y]
            val detectionGeometryX1Data = transposedDetectionGeometries[0][1][y]
            val detectionGeometryX2Data = transposedDetectionGeometries[0][2][y]
            val detectionGeometryX3Data = transposedDetectionGeometries[0][3][y]
            val detectionRotationAngleData = transposedDetectionGeometries[0][4][y]

            for (x in 0 until transposedDetectionScores[0][0][0].size) {
                if (detectionScoreData[x] < 0.5) {
                    continue
                }

                val offsetX = x * 4.0
                val offsetY = y * 4.0

                val h = detectionGeometryX0Data[x] + detectionGeometryX2Data[x]
                val w = detectionGeometryX1Data[x] + detectionGeometryX3Data[x]

                val angle = detectionRotationAngleData[x]
                val cos = Math.cos(angle.toDouble())
                val sin = Math.sin(angle.toDouble())

                val offset = Point(
                    offsetX + cos * detectionGeometryX1Data[x] + sin * detectionGeometryX2Data[x],
                    offsetY - sin * detectionGeometryX1Data[x] + cos * detectionGeometryX2Data[x]
                )
                val p1 = Point(-sin * h + offset.x, -cos * h + offset.y)
                val p3 = Point(-cos * w + offset.x, sin * w + offset.y)
                val center = Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y))

                val textDetection = RotatedRect(
                    center,
                    Size(w.toDouble(), h.toDouble()),
                    (-1 * angle * 180.0 / Math.PI)
                )
                detRotatedRects.add(textDetection)
                detConfidences.add(detectionScoreData[x])
            }
        }

        val detConfidencesMat = MatOfFloat(vector_float_to_Mat(detConfidences))
        val boundingBoxesMat = MatOfRotatedRect(vector_RotatedRect_to_Mat(detRotatedRects))

        NMSBoxesRotated(
            boundingBoxesMat,
            detConfidencesMat,
            detectionConfidenceThreshold,
            detectionNMSThreshold,
            indicesMat
        )

        return Pair(boundingBoxesMat, indicesMat)
    }

    private fun recognizeTexts(
        data: Bitmap,
        boundingBoxesMat: MatOfRotatedRect,
        indicesMat: MatOfInt
    ): Bitmap {
        val bitmapWithBoundingBoxes = data.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmapWithBoundingBoxes)
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 10.toFloat()
        paint.color = Color.GREEN

        val ratioHeight = data.height.toFloat() / detectionImageHeight
        val ratioWidth = data.width.toFloat() / detectionImageWidth

        for (i in indicesMat.toArray()) {
            val boundingBox = boundingBoxesMat.toArray()[i]
            val targetVertices = ArrayList<Point>()
            targetVertices.add(Point(0.toDouble(), (recognitionImageHeight - 1).toDouble()))
            targetVertices.add(Point(0.toDouble(), 0.toDouble()))
            targetVertices.add(Point((recognitionImageWidth - 1).toDouble(), 0.toDouble()))
            targetVertices.add(Point((recognitionImageWidth - 1).toDouble(), (recognitionImageHeight - 1).toDouble()))
            val srcVertices = ArrayList<Point>()
            val boundingBoxPointsMat = Mat()
            boxPoints(boundingBox, boundingBoxPointsMat)
            for (j in 0 until 4) {
                srcVertices.add(
                    Point(
                        boundingBoxPointsMat.get(j, 0)[0] * ratioWidth,
                        boundingBoxPointsMat.get(j, 1)[0] * ratioHeight
                    )
                )
                if (j != 0) {
                    canvas.drawLine(
                        (boundingBoxPointsMat.get(j, 0)[0] * ratioWidth).toFloat(),
                        (boundingBoxPointsMat.get(j, 1)[0] * ratioHeight).toFloat(),
                        (boundingBoxPointsMat.get(j - 1, 0)[0] * ratioWidth).toFloat(),
                        (boundingBoxPointsMat.get(j - 1, 1)[0] * ratioHeight).toFloat(),
                        paint
                    )
                }
            }
            canvas.drawLine(
                (boundingBoxPointsMat.get(0, 0)[0] * ratioWidth).toFloat(),
                (boundingBoxPointsMat.get(0, 1)[0] * ratioHeight).toFloat(),
                (boundingBoxPointsMat.get(3, 0)[0] * ratioWidth).toFloat(),
                (boundingBoxPointsMat.get(3, 1)[0] * ratioHeight).toFloat(),
                paint
            )

            val srcVerticesMat = MatOfPoint2f(srcVertices[0], srcVertices[1], srcVertices[2], srcVertices[3])
            val targetVerticesMat = MatOfPoint2f(targetVertices[0], targetVertices[1], targetVertices[2], targetVertices[3])
            val rotationMatrix = getPerspectiveTransform(srcVerticesMat, targetVerticesMat)
            val recognitionBitmapMat = Mat()
            val srcBitmapMat = Mat()
            bitmapToMat(data, srcBitmapMat)
            warpPerspective(
                srcBitmapMat,
                recognitionBitmapMat,
                rotationMatrix,
                Size(recognitionImageWidth.toDouble(), recognitionImageHeight.toDouble())
            )

            val recognitionBitmap = createEmptyBitmap(
                recognitionImageWidth,
                recognitionImageHeight,
                0,
                Bitmap.Config.ARGB_8888
            )
            matToBitmap(recognitionBitmapMat, recognitionBitmap)

            val recognitionTensorImage = bitmapToTensorImageForRecognition(
                recognitionBitmap,
                recognitionImageWidth,
                recognitionImageHeight,
                recognitionImageMean,
                recognitionImageStd
            )

            recognitionResult.rewind()
            trOcrRecognizer.run(recognitionTensorImage.buffer, recognitionResult)

            var recognizedText = ""
            for (k in 0 until recognitionModelOutputSize) {
                val alphabetIndex = recognitionResult.getInt(k * 8)
                if (alphabetIndex in 0..alphabets.length - 1) recognizedText += alphabets[alphabetIndex]
            }
            Log.d("Recognition result:", recognizedText)
            if (recognizedText != "") {
                currentOcrResults.add(recognizedText) // Add to current results
                ocrResults[recognizedText] = getRandomColor()
            }
        }
        return bitmapWithBoundingBoxes
    }

    private fun performLLMInference(ocrText: String = "") {
        val userInput = if (ocrText.isEmpty()) inputEditText.text.toString() else ocrText
        if (userInput.isNotEmpty()) {
            chatMessages.add(ChatMessage(userInput, true))
            chatAdapter.notifyItemInserted(chatMessages.size - 1)
            chatRecyclerView.scrollToPosition(chatMessages.size - 1)

            inputEditText.text.clear()

            val timeSource = TimeSource.Monotonic
            val mark1 = timeSource.markNow()

            val systemPrompt = if (ocrText.isNotEmpty()) {
                "Return the following input in JSON format where the first word is a key, the second word is its value, the third word is a key value. Input:\n"
            } else {
                "You are a helpful assistant. Respond to the user's query."
            }

            val conversationContext = conversationHistory.joinToString("\n")
            val inputPrompt = "$systemPrompt\n$conversationContext\nUser: $userInput\nAssistant:"
            Log.d("LLM", "Input prompt: $inputPrompt")

            val result = llmInference.generateResponse(inputPrompt)
            Log.d("LLM", "LLM response: $result")

            conversationHistory.add("User: $userInput")
            conversationHistory.add("Assistant: $result")

            val mark2 = timeSource.markNow()
            val elapsed = mark2 - mark1
            Log.d("Time", "Elapsed: $elapsed")

            val promptTokenSize = llmInference.sizeInTokens(inputPrompt)
            Log.d("Tokens", "Input prompt token size: $promptTokenSize")

            val basicTokenSpeed = promptTokenSize / elapsed.inWholeSeconds
            Log.d("Time", "Crude Token/s : $basicTokenSpeed")

            chatMessages.add(ChatMessage(result, false))
            chatAdapter.notifyItemInserted(chatMessages.size - 1)
            chatRecyclerView.scrollToPosition(chatMessages.size - 1)
        }
    }

    fun getRandomColor(): Int {
        val random = Random()
        return Color.argb(
            128,
            (255 * random.nextFloat()).toInt(),
            (255 * random.nextFloat()).toInt(),
            (255 * random.nextFloat()).toInt()
        )
    }

    fun bitmapToTensorImageForRecognition(
        bitmapIn: Bitmap,
        width: Int,
        height: Int,
        mean: Float,
        std: Float
    ): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
            .add(TransformToGrayscaleOp())
            .add(NormalizeOp(mean, std))
            .build()
        var tensorImage = TensorImage(DataType.FLOAT32)

        tensorImage.load(bitmapIn)
        tensorImage = imageProcessor.process(tensorImage)

        return tensorImage
    }

    fun createEmptyBitmap(
        imageWidth: Int,
        imageHeight: Int,
        color: Int = 0,
        imageConfig: Bitmap.Config = Bitmap.Config.RGB_565
    ): Bitmap {
        val ret = Bitmap.createBitmap(imageWidth, imageHeight, imageConfig)
        if (color != 0) {
            ret.eraseColor(color)
        }
        return ret
    }

    private fun preprocessImage(bitmap: Bitmap, width: Int, height: Int, mean: FloatArray, std: FloatArray): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(mean, std))
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)
        return tensorImage
    }
}