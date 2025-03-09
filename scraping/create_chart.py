import pyodbc
import numpy as np
import matplotlib.pyplot as plt

# Database connection parameters (same as in the scraper script)
DB_SERVER = "193.85.203.188"
DB_NAME = "boska"
DB_USER = "boska"
DB_PASSWORD = "123456"

def connect_to_db():
    """Establish a connection to the database."""
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD}"
    try:
        conn = pyodbc.connect(conn_str)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def analyze_articles():
    """Analyze and visualize article data."""
    conn = connect_to_db()
    if not conn:
        return

    try:
        # Query to get article counts by source
        cursor = conn.cursor()
        cursor.execute("""
            SELECT SourceName, 
                   COUNT(*) as ArticleCount, 
                   AVG(ArticleLength) as AvgLength, 
                   AVG(WordCount) as AvgWordCount
            FROM Articles
            GROUP BY SourceName
        """)
        
        # Fetch results
        results = cursor.fetchall()
        
        # Prepare data for visualization
        sources = [row.SourceName for row in results]
        article_counts = [row.ArticleCount for row in results]
        avg_lengths = [row.AvgLength for row in results]
        avg_word_counts = [row.AvgWordCount for row in results]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Article Count Bar Chart
        ax1.bar(sources, article_counts, color='skyblue')
        ax1.set_title('Number of Articles by News Source')
        ax1.set_xlabel('News Source')
        ax1.set_ylabel('Number of Articles')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average Article Length Bar Chart
        ax2.bar(sources, avg_lengths, color='lightgreen')
        ax2.set_title('Average Article Length by News Source')
        ax2.set_xlabel('News Source')
        ax2.set_ylabel('Average Article Length (Characters)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('news_sources_analysis.png')
        plt.close()
        
        # Print some additional insights
        print("\nNews Sources Analysis:")
        for source, count, length, words in zip(sources, article_counts, avg_lengths, avg_word_counts):
            print(f"{source}:")
            print(f"  - Total Articles: {count}")
            print(f"  - Avg Article Length: {length:.2f} characters")
            print(f"  - Avg Word Count: {words:.2f} words")
        
    except Exception as e:
        print(f"Error analyzing data: {e}")
    finally:
        conn.close()

# Run the analysis
if __name__ == "__main__":
    analyze_articles()