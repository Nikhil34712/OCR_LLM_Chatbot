package com.example.ocr_phi

data class ChatMessage(val content: Any, val isUser: Boolean, val isImage: Boolean = false)