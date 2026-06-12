import os
import random
import pandas as pd
import requests

# Targets
DATASET_DIR = "c:/Users/shash/Documents/FakeOrRealNews/dataset"
FAKE_PATH = os.path.join(DATASET_DIR, "fake.csv")
REAL_PATH = os.path.join(DATASET_DIR, "real.csv")

# URL for downloading fake_or_real_news.csv (approx 6300 articles)
DATA_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"

# Representative sources for enrichment
REAL_SOURCES = ["Reuters", "Associated Press", "BBC News", "CNN", "The New York Times", "NPR", "Washington Post"]
FAKE_SOURCES = ["Political Insider", "Freedom Daily", "Activist Post", "The Onion", "Neon Nettle", "Infowars", "Daily Buzz"]

# Representative subjects
SUBJECTS = ["Politics", "World News", "Middle East", "US News", "Government News", "Science & Tech", "Health"]

def download_dataset():
    """Tries to download the dataset from GitHub."""
    print(f"Attempting to download dataset from: {DATA_URL} ...")
    try:
        # Stream download in chunks to avoid memory spikes
        response = requests.get(DATA_URL, timeout=30, stream=True)
        response.raise_for_status()
        
        temp_csv = "temp_dataset.csv"
        with open(temp_csv, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        df = pd.read_csv(temp_csv)
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
        print(f"Successfully downloaded dataset with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Download failed: {e}. Moving to synthetic generation...")
        return None

def generate_synthetic_data(num_samples=1000):
    """Generates realistic synthetic news dataset when download fails."""
    print(f"Generating {num_samples} synthetic news articles...")
    
    real_headlines = [
        "Congress Passes Landmark Environmental Regulation Bill",
        "Global Markets Rally Amid Favorable Inflation Data",
        "New Breakthrough in Fusion Energy Announced by National Lab",
        "Healthcare Reform Proposal Debated in Senate Committee",
        "SpaceX Successfully Launches Next-Generation Satellite",
        "European Union Agrees on New Tech Regulations for AI",
        "Federal Reserve Decides to Hold Interest Rates Steady",
        "Scientists Discover New Species of Deep Sea Marine Life",
        "City Council Approves New Affordable Housing Initiative",
        "WHO Outlines Global Response Strategy for Emerging Virus Variants"
    ]
    
    fake_headlines = [
        "SHOCKING: Hidden Microchips Found in Local Water Supply!",
        "Leaked Video Confirms Moon Landing Was Filmed in Hollywood Studio",
        "BREAKING: Celebrity Secretly Replaced by Clone, Source Claims",
        "Secret Society Controls Elections from Underground Swiss Bunker",
        "ALERT: Common Vegetable Classified as Dangerous Biohazard by Government",
        "Miracle Fruit Cures All Diseases in 24 Hours, Medical Elites Banned It!",
        "UN to Re-classify Earth as Flat starting Next Month",
        "Billionaire Buys Entire Town to Build Private Dinosaurs Park",
        "Local Mayor Revealed to be Reptilian Shape-shifter in Caught-on-Camera Video",
        "Mysterious Portal Discovered in Desert, Aliens Seen Entering"
    ]
    
    real_bodies = [
        "In a historic session, members of Congress yesterday voted in favor of a sweeping environmental package aimed at reducing carbon emissions and providing federal subsidies for renewable energy. The bipartisan bill was hailed by advocates as a critical step forward. Critics, however, argue that the compliance costs could hurt manufacturing sectors.",
        "Stocks closed higher today as positive consumer index data suggested inflation pressures may be cooling faster than economists predicted. The Dow Jones Industrial Average rose nearly two percent, led by tech and healthcare gains. Analysts attribute this rebound to investors gaining confidence in economic stability.",
        "Researchers at the National Ignition Facility announced a major milestone in fusion energy research, achieving a net energy gain for the second time using advanced lasers. The experiment yielded more energy from fusion than the laser energy put in. If scaled, fusion could provide clean, limitless energy.",
        "The Senate Finance Committee held a contentious hearing today over the proposed healthcare bill, with members debating the cost projections and coverage expansions. The bill aims to lower prescription drug prices and cap out-of-pocket expenses. Opponents expressed concern over potential budget deficits.",
        "A Falcon Heavy rocket lifted off from Kennedy Space Center earlier today, carrying a high-capacity communications satellite into orbit. The payload was deployed successfully forty minutes post-launch, securing broadband coverage for remote regions. Both side boosters returned and landed in unison.",
        "The European Parliament has drafted a comprehensive regulatory framework governing the deployment of generative AI technologies. Under the new rules, developers must audit datasets and register foundational models. Fines for non-compliance could reach up to seven percent of global revenue."
    ]
    
    fake_bodies = [
        "You won't believe what they are hiding! Sources close to the administration have confirmed that the water supply has been spiked with tiny tracking microchips. The government is using this to monitor citizen movements. Officials are denying the claims, but citizens are reporting weird metallic tastes.",
        "A top-secret video leaked online proves that the 1969 moon landing was entirely staged. The footage shows astronauts walking on a sound stage in California with directors shouting commands. NASA has scrambled to delete the videos, but internet investigators have archived copies.",
        "A whistleblower claims a famous hollywood star has been missing for years and was replaced by a lookalike robotic clone. The clone reportedly malfunctioned during a live interview, revealing mechanical components. Hollywood executives are trying to cover up the scandal.",
        "A hidden document reveals that a secret cabal of global elites meets annually in an underground Swiss bunker to decide election outcomes across the globe. Using mind control frequencies, they influence voting machines. The public is kept in the dark by mainstream media outlets.",
        "Health officials are panic-stricken after a secret report leaked online warning that common carrots are actually genetically engineered biohazards. The report claims they were designed to collect DNA samples. Activists are demanding an immediate ban on all root vegetables.",
        "A breakthrough organic discovery—a rare Amazonian berry—has been shown to eliminate all chronic ailments in less than a day. Despite this, medical associations have banned its sale to protect pharmaceutical profits. Learn the truth before this webpage is taken down!"
    ]
    
    data = []
    for i in range(num_samples):
        is_real = i % 2 == 1
        label = "REAL" if is_real else "FAKE"
        
        if is_real:
            title = random.choice(real_headlines) + f" (Report {i})"
            text = random.choice(real_bodies) + f" The Associated Press contributed to this report."
            source = random.choice(REAL_SOURCES)
        else:
            title = random.choice(fake_headlines) + f" !!! (MUST READ {i})"
            text = random.choice(fake_bodies) + f" Don't trust the mainstream media, share this now!"
            source = random.choice(FAKE_SOURCES)
            
        subject = random.choice(SUBJECTS)
        date = f"June {random.randint(1, 28)}, 2024"
        
        data.append({
            "title": title,
            "text": text,
            "label": label,
            "source": source,
            "subject": subject,
            "date": date
        })
        
    return pd.DataFrame(data)

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # 1. Download or Generate
    df = download_dataset()
    if df is None:
        df = generate_synthetic_data(1200)
    else:
        # Standard fake_or_real_news.csv has columns: [Unnamed: 0, title, text, label]
        # Let's clean and enrich it
        if "label" not in df.columns and "Label" in df.columns:
            df = df.rename(columns={"Label": "label"})
        
        # Ensure we have label, title, text
        if "label" not in df.columns:
            # Maybe it uses 0 and 1
            if "class" in df.columns:
                df['label'] = df['class'].apply(lambda x: "FAKE" if x == 0 or x == '0' else "REAL")
            else:
                # Add default label for testing
                df['label'] = ["REAL" if i % 2 == 0 else "FAKE" for i in range(len(df))]
                
        # Fill missing values
        df['title'] = df['title'].fillna("No Title")
        df['text'] = df['text'].fillna("No Content")
        
        # Normalize labels to uppercase FAKE/REAL
        df['label'] = df['label'].astype(str).str.upper()
        df['label'] = df['label'].apply(lambda x: "REAL" if "REAL" in x or "TRUE" in x or x == '1' else "FAKE")
        
        # Enrich with subjects and sources to mock ISOT/WELFake structure
        print("Enriching dataset with sources and subjects...")
        sources = []
        subjects = []
        dates = []
        for index, row in df.iterrows():
            subj = random.choice(SUBJECTS)
            dt = f"June {random.randint(1, 28)}, 2024"
            if row['label'] == "REAL":
                src = random.choice(REAL_SOURCES)
            else:
                src = random.choice(FAKE_SOURCES)
            sources.append(src)
            subjects.append(subj)
            dates.append(dt)
            
        df['source'] = sources
        df['subject'] = subjects
        df['date'] = dates

    # 2. Save split datasets
    fake_df = df[df['label'] == 'FAKE'][['title', 'text', 'source', 'subject', 'date']]
    real_df = df[df['label'] == 'REAL'][['title', 'text', 'source', 'subject', 'date']]
    
    # Save to CSV
    fake_df.to_csv(FAKE_PATH, index=False)
    real_df.to_csv(REAL_PATH, index=False)
    
    print(f"Dataset Split Completed:")
    print(f"  Saved {len(fake_df)} fake articles to {FAKE_PATH}")
    print(f"  Saved {len(real_df)} real articles to {REAL_PATH}")

if __name__ == "__main__":
    main()
