use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("backend init failed: {0}")]
    Backend(String),
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("context creation failed: {0}")]
    Context(String),
    #[error("tokenization failed: {0}")]
    Tokenize(String),
    #[error("chat template failed: {0}")]
    ChatTemplate(String),
    #[error("decode failed: {0}")]
    Decode(String),
    /// The request carried images but the model was loaded without a projector.
    #[error(
        "{images} image(s) attached, but this model has no vision projector (no mmproj loaded)"
    )]
    VisionUnavailable { images: usize },
    /// Image encoding or multimodal prompt assembly failed.
    #[error("vision failed: {0}")]
    Vision(String),
    /// Embedding a text failed, usually because the model has no pooling type.
    #[error("embedding failed: {0}")]
    Embedding(String),
    /// The loaded model's class cannot do what the request asked of it.
    ///
    /// Refusing here is what keeps an encoder from being sampled for text and a
    /// chat model from being asked for a pooled vector — neither has the output
    /// the other would read.
    #[error("this is a {class} model, which cannot {what}")]
    Unsupported { class: String, what: String },
    /// A request parameter was outside the range this engine can honour.
    #[error("{0}")]
    InvalidParameter(String),
    /// Reranking failed, usually because the model has no rank head.
    #[error("rerank failed: {0}")]
    Rerank(String),
    /// The prompt does not leave room to generate.
    #[error("the prompt needs {used} tokens but the context is only {n_ctx}")]
    ContextFull { used: usize, n_ctx: usize },
    /// The caller stopped listening, so there is no point generating further.
    #[error("client disconnected")]
    ClientGone,
    /// The caller asked to stop before the model had finished.
    #[error("generation cancelled")]
    Cancelled,
    #[error("worker thread is gone")]
    WorkerGone,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, InferError>;
