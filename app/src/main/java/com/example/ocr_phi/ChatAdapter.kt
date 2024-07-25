package com.example.ocr_phi

class ChatAdapter(private val messages: List<ChatMessage>) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    private val VIEW_TYPE_USER = 1
    private val VIEW_TYPE_ASSISTANT = 2
    private val VIEW_TYPE_IMAGE = 3

    override fun getItemViewType(position: Int): Int {
        val message = messages[position]
        return when {
            message.isImage -> VIEW_TYPE_IMAGE
            message.isUser -> VIEW_TYPE_USER
            else -> VIEW_TYPE_ASSISTANT
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return when (viewType) {
            VIEW_TYPE_USER -> MessageViewHolder(LayoutInflater.from(parent.context).inflate(R.layout.item_message_user, parent, false))
            VIEW_TYPE_ASSISTANT -> MessageViewHolder(LayoutInflater.from(parent.context).inflate(R.layout.item_message_bot, parent, false))
            VIEW_TYPE_IMAGE -> ImageMessageViewHolder(LayoutInflater.from(parent.context).inflate(R.layout.item_image_message, parent, false))
            else -> throw IllegalArgumentException("Invalid view type")
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        when (holder) {
            is MessageViewHolder -> {
                if (message.content is String) {
                    holder.bind(message.content)
                } else {
                    holder.bind("Unsupported message type")
                }
            }
            is ImageMessageViewHolder -> {
                if (message.content is Bitmap) {
                    holder.bind(message.content)
                } else {
                    holder.bind(BitmapFactory.decodeResource(holder.itemView.resources, android.R.drawable.ic_menu_report_image))
                }
            }
        }
    }

    override fun getItemCount() = messages.size

    class MessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val messageTextView: TextView = itemView.findViewById(R.id.messageText)

        fun bind(message: String) {
            messageTextView.text = message
        }
    }

    class ImageMessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageView: ImageView = itemView.findViewById(R.id.imageMessageView)

        fun bind(bitmap: Bitmap) {
            imageView.setImageBitmap(bitmap)
        }
    }
}