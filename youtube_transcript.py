from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import Json

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

def get_metadata(video_url):
    with YoutubeDL() as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info

def segment_transcript(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    transcript_text = "\n".join([t["text"] for t in transcript])
    segments = text_splitter.split_text(transcript_text)

    # Return each segment with its start and end times
    return [{'start': transcript[i]['start'], 'end': transcript[i]['start'] + 500, 'text': segment}
            for i, segment in enumerate(segments)]

def annotate_segments(segments):
    # Use Hugging Face summarization model
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

    # Process each segment by summarizing it
    annotated_segments = []
    for segment in segments:
        result = summarizer(segment["text"], max_length=100, min_length=20, do_sample=False)
        annotated_segments.append(result[0]["summary_text"])
    return annotated_segments

def get_embeddings(segments):
    # Use SentenceTransformer for embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return [embed_model.encode(segment["text"]) for segment in segments]

# Database setup
def setup_database():
    conn = psycopg2.connect(
        dbname="postgres",
        user="mirapanigrahy",
        password="password",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Adjust vector size to 384 for Hugging Face embeddings
    cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS video_segments (
        id SERIAL PRIMARY KEY,
        video_id TEXT,
        start_time FLOAT,
        end_time FLOAT,
        segment TEXT,
        embedding VECTOR(384)  -- Adjust vector size for Hugging Face embeddings
    );
    """)
    conn.commit()

    return conn, cursor

# Function to process a list of video URLs
def process_videos(video_urls):
    for video_url in video_urls:
        # Extract video ID from the URL
        video_id = video_url.split("v=")[-1]  # For example, extract "v=XXXX" from the URL
        print(f"Processing video: {video_id}")

        # Get transcript and metadata
        transcript = get_transcript(video_id)
        if transcript:
            # Segment the transcript
            segments = segment_transcript(transcript)
            
            # Get embeddings for each segment
            embeddings = get_embeddings(segments)
            
            # Insert the data into the database
            store_embeddings(conn, cursor, video_id, segments, embeddings)
        else:
            print(f"Skipping video {video_id} due to missing transcript.")

# Store embeddings in database
def store_embeddings(conn, cursor, video_id, segments, embeddings):
    for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
        # Convert the numpy array to a list of floats explicitly
        embedding_list = embedding.tolist()  # Convert embedding to list
        embedding_list = [float(e) for e in embedding_list]  # Convert each element to float
        
        cursor.execute("""
        INSERT INTO video_segments (video_id, start_time, end_time, segment, embedding)
        VALUES (%s, %s, %s, %s, %s);
        """, (video_id, segment['start'], segment['end'], segment['text'], embedding_list))  # Ensure embedding is a list of floats
    conn.commit()

def query_video_segments(conn, cursor, query, top_k=10):
    # Create embedding for the query text using Hugging Face model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode([query])[0]  # Get the embedding for the query text

    # Convert query embedding to a list
    query_embedding = query_embedding.tolist()  # Convert to list
    query_embedding = [float(e) for e in query_embedding]  # Ensure each element is a float

    # Use PGVector for similarity search, cast to vector type explicitly
    cursor.execute("""
    SELECT DISTINCT ON (video_id) id, video_id, start_time, end_time, segment, embedding <=> %s::vector AS similarity
    FROM video_segments
    ORDER BY video_id, similarity
    LIMIT %s;
    """, (query_embedding, top_k))


    results = cursor.fetchall()
    return results

# Main execution logic
if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sample_text = "Test embedding"
    embedding = model.encode(sample_text)
    print(f"Embedding size: {len(embedding)}")  # Should print 384

    # Database connection
    conn, cursor = setup_database()

    # Add videos to database
    video_urls = ["https://www.youtube.com/watch?v=rL8X2mlNHPM",
    "https://www.youtube.com/watch?v=DuDz6B4cqVc",
    "https://www.youtube.com/watch?v=ZoqMiFKspAA",
    "https://www.youtube.com/watch?v=HjneAhCy2N4",
    "https://www.youtube.com/watch?v=6-tKOHICqrI",
    "https://www.youtube.com/watch?v=26QPDBe-NB8",
    "https://www.youtube.com/watch?v=pVzRTmdd9j0",
    "https://www.youtube.com/watch?v=GjNp0bBrjmU",
    "https://www.youtube.com/watch?v=ACsLvXuaKxw",
    "https://www.youtube.com/watch?v=O753uuutqH8",
    "https://www.youtube.com/watch?v=iIxZrYzJJ7I",
    "https://www.youtube.com/watch?v=rrB13utjYV4",
    "https://www.youtube.com/watch?v=__iKSnQXe_o"]
  
    process_videos(video_urls)

    # Query for a topic
    user_query = input("Enter your query: ")
    results = query_video_segments(conn, cursor, user_query)

    # Display query results
    print("Query Results:")
    for result in results:
        print(f"Video ID: {result[1]}, Start Time: {result[2]}, End Time: {result[3]}")
        print(f"Segment: {result[4]}")
        print("------")

    # Close database connection
    cursor.close()  # Close cursor after all database operations
    conn.close()  # Close the connection after all operations

