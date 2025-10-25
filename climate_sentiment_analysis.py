#!/usr/bin/env python3
"""
Climate Change Social Media Mining - Sentiment Analysis
Analyzes sentiment of climate change related posts using TextBlob
"""

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Sample climate change related social media posts
# In a real scenario, these would be fetched from Twitter/X API or other social media platforms
sample_posts = [
    "Climate change is the biggest threat to our planet. We need immediate action!",
    "Renewable energy is the future. Solar and wind power can save us.",
    "The effects of global warming are devastating coastal communities.",
    "Scientists warn about rising sea levels due to melting ice caps.",
    "Carbon emissions must be reduced drastically to avoid catastrophe.",
    "Green technology is making great progress in fighting climate change.",
    "Extreme weather events are becoming more frequent due to climate change.",
    "Investing in clean energy will create millions of jobs.",
    "Deforestation is accelerating climate change at an alarming rate.",
    "Youth activists are leading the charge for climate action.",
    "Electric vehicles are helping reduce carbon footprint.",
    "Coral reefs are dying due to ocean acidification.",
    "Sustainable practices are essential for our planet's future.",
    "Climate policies need to be more aggressive to meet targets.",
    "Innovation in green technology gives hope for the future.",
    "Polar bears are losing their habitat due to melting ice.",
    "Countries must work together to combat global warming.",
    "Climate change denial is dangerous and anti-scientific.",
    "Renewable energy is now cheaper than fossil fuels.",
    "Every individual can make a difference in fighting climate change."
]

# Create timestamps for posts (simulated)
timestamps = pd.date_range(start='2024-01-01', periods=len(sample_posts), freq='D')

class ClimateS entimentAnalyzer:
    """Analyze sentiment of climate change related social media posts"""
    
    def __init__(self, posts, timestamps=None):
        self.posts = posts
        self.timestamps = timestamps if timestamps is not None else [datetime.now()] * len(posts)
        self.results = None
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text using TextBlob"""
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def analyze_all_posts(self):
        """Analyze sentiment for all posts"""
        results = []
        
        for i, post in enumerate(self.posts):
            sentiment_data = self.analyze_sentiment(post)
            results.append({
                'post': post,
                'timestamp': self.timestamps[i],
                'polarity': sentiment_data['polarity'],
                'subjectivity': sentiment_data['subjectivity'],
                'sentiment': sentiment_data['sentiment']
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def visualize_results(self):
        """Create visualizations of sentiment analysis results"""
        if self.results is None:
            print("No results to visualize. Run analyze_all_posts() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Climate Change Social Media Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution
        sentiment_counts = self.results['sentiment'].value_counts()
        colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
        sentiment_colors = [colors[sent] for sent in sentiment_counts.index]
        
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                       colors=sentiment_colors, startangle=90)
        axes[0, 0].set_title('Sentiment Distribution', fontweight='bold')
        
        # 2. Polarity Distribution
        axes[0, 1].hist(self.results['polarity'], bins=15, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(self.results['polarity'].mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {self.results["polarity"].mean():.3f}')
        axes[0, 1].set_xlabel('Polarity Score', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Polarity Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Polarity vs Subjectivity Scatter
        scatter_colors = self.results['sentiment'].map(colors)
        axes[1, 0].scatter(self.results['polarity'], self.results['subjectivity'],
                          c=scatter_colors, s=100, alpha=0.6, edgecolors='black')
        axes[1, 0].set_xlabel('Polarity', fontweight='bold')
        axes[1, 0].set_ylabel('Subjectivity', fontweight='bold')
        axes[1, 0].set_title('Polarity vs Subjectivity', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Create legend for scatter plot
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[s], label=s) for s in ['Positive', 'Neutral', 'Negative']]
        axes[1, 0].legend(handles=legend_elements, loc='best')
        
        # 4. Sentiment Over Time
        if len(self.results) > 1:
            sentiment_numeric = self.results['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
            axes[1, 1].plot(self.results['timestamp'], self.results['polarity'],
                          marker='o', linestyle='-', color='#9b59b6', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('Date', fontweight='bold')
            axes[1, 1].set_ylabel('Polarity Score', fontweight='bold')
            axes[1, 1].set_title('Sentiment Polarity Over Time', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('climate_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'climate_sentiment_analysis.png'")
        plt.show()
    
    def print_statistics(self):
        """Print summary statistics"""
        if self.results is None:
            print("No results available. Run analyze_all_posts() first.")
            return
        
        print("\n" + "="*60)
        print("CLIMATE CHANGE SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nTotal Posts Analyzed: {len(self.results)}")
        print(f"\nSentiment Distribution:")
        print(self.results['sentiment'].value_counts())
        print(f"\nAverage Polarity: {self.results['polarity'].mean():.4f}")
        print(f"Average Subjectivity: {self.results['subjectivity'].mean():.4f}")
        print(f"\nMost Positive Post:")
        most_positive = self.results.loc[self.results['polarity'].idxmax()]
        print(f"  - Polarity: {most_positive['polarity']:.4f}")
        print(f"  - Text: {most_positive['post']}")
        print(f"\nMost Negative Post:")
        most_negative = self.results.loc[self.results['polarity'].idxmin()]
        print(f"  - Polarity: {most_negative['polarity']:.4f}")
        print(f"  - Text: {most_negative['post']}")
        print("\n" + "="*60)

def main():
    """Main execution function"""
    print("Starting Climate Change Social Media Sentiment Analysis...")
    print(f"Analyzing {len(sample_posts)} sample posts\n")
    
    # Initialize analyzer
    analyzer = ClimateSentimentAnalyzer(sample_posts, timestamps)
    
    # Analyze all posts
    results_df = analyzer.analyze_all_posts()
    
    # Print statistics
    analyzer.print_statistics()
    
    # Save results to CSV
    results_df.to_csv('climate_sentiment_results.csv', index=False)
    print("\nResults saved to 'climate_sentiment_results.csv'")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_results()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
