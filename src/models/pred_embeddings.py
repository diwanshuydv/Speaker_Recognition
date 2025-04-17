from resemblyzer import preprocess_wav, VoiceEncoder
from qdrant_client import QdrantClient
from collections import Counter


def pred_speaker(wav_path: str, collection_name: str = "speaker_recognition_testing", url: str = "http://localhost:6333"):
    client = QdrantClient(url=url)
    encoder = VoiceEncoder("cuda")
    test_wav = preprocess_wav(wav_path)
    test_embeddings = encoder.embed_utterance(test_wav)
    results = client.search(collection_name, test_embeddings, score_threshold=0.6)
    top_5_speaker_ids = [result.payload["speaker_id"] for result in results]
    speaker_id_counts = Counter(top_5_speaker_ids)
    most_frequent_speaker_id = speaker_id_counts.most_common(1)[0][0]
    return most_frequent_speaker_id
