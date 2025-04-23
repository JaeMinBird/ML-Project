import requests
import pandas as pd
import logging
from datetime import datetime
import os
import time
import random
import json
import re
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'scraper_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ratemyprofessors_scraper')

class RateMyProfessorsScraper:
    def __init__(self):
        self.base_url = "https://www.ratemyprofessors.com"
        self.search_url = "https://www.ratemyprofessors.com/search/professors"
        self.school_name = "Penn State University"
        self.school_id = "758"  # Penn State University ID for RateMyProfessors
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.ratemyprofessors.com/",
        }
        # Maximum retries for web requests
        self.max_retries = 3
        # Delay between retries (in seconds)
        self.retry_delay = 2
        # Track already scraped professor IDs to avoid duplicates
        self.scraped_professor_ids = set()

    def _make_request(self, url, params=None):
        """Make a GET request to the RateMyProfessors website with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Making request to {url} (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.get(url, headers=self.headers, params=params)
                
                # Log response status
                logger.info(f"Response status code: {response.status_code}")
                
                # Check if we got a non-200 response
                if response.status_code != 200:
                    logger.error(f"Website returned error status: {response.status_code}")
                    # Wait before retrying
                    time.sleep(self.retry_delay)
                    continue
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                
            # Add a random delay between retries to avoid rate limiting
            delay = self.retry_delay + random.uniform(0, 2)
            logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            
        logger.error("All request attempts failed")
        return None

    def _test_connection(self):
        """Test the connection to RateMyProfessors website"""
        logger.info("Testing connection to RateMyProfessors website")
        
        response = self._make_request(self.base_url)
        if response and response.status_code == 200:
            logger.info("Connection test successful")
            return True
        else:
            logger.error("Connection test failed")
            return False

    def _extract_relay_store(self, html_content):
        """Extract the RELAY_STORE data structure from the HTML content"""
        try:
            # Look for the script that contains __RELAY_STORE__
            pattern = r'window\.__RELAY_STORE__\s*=\s*({.*?});'
            matches = re.search(pattern, html_content, re.DOTALL)
            
            if matches:
                relay_store_json = matches.group(1)
                # Parse the JSON data
                try:
                    relay_store = json.loads(relay_store_json)
                    return relay_store
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse RELAY_STORE JSON: {e}")
            else:
                logger.error("Could not find RELAY_STORE in HTML")
            
            return None
        except Exception as e:
            logger.error(f"Error extracting RELAY_STORE: {e}")
            return None

    def _extract_top_tags(self, professor_url):
        """Extract top tags from a professor's page using BeautifulSoup"""
        logger.info(f"Extracting top tags from {professor_url}")
        
        response = self._make_request(professor_url)
        if not response:
            logger.error("Failed to get professor page for tag extraction")
            return []
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the tags container (based on the example page HTML structure)
            # The tags are in spans inside a div with a class related to tags
            tags_div = soup.find('div', string=lambda s: s and "Top Tags" in s)
            if tags_div:
                # Find the container with the actual tags (usually a sibling or child element)
                tags_container = tags_div.find_next('div')
                
                if tags_container:
                    # Extract all span elements which typically contain the tags
                    tag_spans = tags_container.find_all('span')
                    tags = [span.text.strip() for span in tag_spans if span.text.strip()]
                    
                    logger.info(f"Found {len(tags)} top tags: {tags}")
                    return tags
                else:
                    logger.warning("Tags container not found")
            else:
                logger.warning("Top Tags section not found on professor page")
            
            # Fallback: return empty list if no tags found
            return []
        except Exception as e:
            logger.error(f"Error extracting top tags: {e}")
            return []

    def search_professors(self, school="Penn State University", page=1):
        """Search for professors at the specified school"""
        logger.info(f"Searching for professors at {school} (page {page})")
        
        # Updated search parameters
        search_params = {
            "q": school,
            "sid": self.school_id,  # School ID for Penn State
            "page": page
        }
        
        response = self._make_request(f"{self.search_url}", params=search_params)
        if not response:
            logger.error(f"Failed to get search results for {school}")
            return [], False
        
        relay_store = self._extract_relay_store(response.text)
        if not relay_store:
            logger.error("Could not extract professor data from response")
            return [], False
        
        # Extract professor data from the relay store
        professors = []
        
        # Find professor nodes in the relay store
        for key, value in relay_store.items():
            if isinstance(value, dict) and value.get("__typename") == "Teacher":
                try:
                    professor_id = value.get("legacyId")
                    name = value.get("firstName", "") + " " + value.get("lastName", "")
                    department = value.get("department", "Unknown")
                    avg_rating = value.get("avgRating", 0.0)
                    
                    # Create the professor URL
                    professor_url = f"{self.base_url}/professor/{professor_id}"
                    
                    professors.append({
                        'name': name.strip(),
                        'department': department,
                        'avg_rating': avg_rating,
                        'url': professor_url,
                        'id': professor_id
                    })
                except Exception as e:
                    logger.error(f"Error parsing professor data: {e}")
        
        # Check if there are more results (next page)
        # We need to look at the pagination info in the relay store
        has_next_page = False
        for key, value in relay_store.items():
            if isinstance(value, dict) and value.get("__typename") == "SearchProfessorsConnection":
                page_info = value.get("pageInfo", {})
                has_next_page = page_info.get("hasNextPage", False)
                break
        
        logger.info(f"Found {len(professors)} professors on page {page}")
        return professors, has_next_page

    def get_professor_reviews(self, professor_url, professor_id):
        """Get reviews for a specific professor and their top tags"""
        if not professor_url:
            logger.error("No professor URL provided")
            return []
        
        logger.info(f"Getting reviews from {professor_url}")
        
        # First, extract the professor's top tags
        top_tags = self._extract_top_tags(professor_url)
        
        response = self._make_request(professor_url)
        if not response:
            logger.error("Failed to get professor page")
            return []
        
        relay_store = self._extract_relay_store(response.text)
        if not relay_store:
            logger.error("Could not extract review data from response")
            return []
        
        reviews = []
        
        # Find review nodes in the relay store
        for key, value in relay_store.items():
            if isinstance(value, dict) and value.get("__typename") == "Rating":
                try:
                    # Extract review data
                    date = value.get("date", "Unknown")
                    class_name = value.get("class", "Unknown")
                    comment = value.get("comment", "")
                    rating = value.get("helpfulRating", 0.0)
                    
                    # Extract tags from the "tags" field if available
                    tags = value.get("tags", [])
                    
                    # Use the top_tags from the professor's profile if review tags are empty
                    if not tags and top_tags:
                        tags = top_tags
                    
                    # Get additional metrics
                    difficulty = value.get("difficultyRating", 0.0)
                    would_take_again = value.get("wouldTakeAgain", None)
                    
                    reviews.append({
                        'date': date,
                        'class': class_name,
                        'text': comment,
                        'rating': rating,
                        'tags': tags,
                        'professor_top_tags': top_tags,  # Store top tags separately
                        'difficulty': difficulty,
                        'would_take_again': would_take_again
                    })
                except Exception as e:
                    logger.error(f"Error parsing review data: {e}")
        
        logger.info(f"Found {len(reviews)} reviews for professor")
        return reviews

    def scrape_penn_state_reviews(self, max_professors=50, max_reviews_per_professor=20):
        """Scrape reviews for Penn State professors and save as CSV"""
        logger.info("Starting to scrape Penn State professor reviews")
        
        # Test the connection first
        if not self._test_connection():
            logger.error("Failed to connect to RateMyProfessors website. Aborting.")
            return None
            
        all_reviews = []
        professors_count = 0
        
        # We'll scrape up to max_professors professors, randomizing the order to avoid repetition
        max_pages = 10  # Maximum pages to search through
        all_professors = []
        page = 1
        has_next_page = True
        
        # First, collect all professors from multiple pages
        while has_next_page and page <= max_pages:
            professors, has_next_page = self.search_professors(school="Penn State University", page=page)
            
            if professors:
                all_professors.extend(professors)
                
            page += 1
            
            # Add a delay between pages to avoid rate limiting
            if has_next_page:
                time.sleep(random.uniform(2, 5))
        
        logger.info(f"Collected information for {len(all_professors)} professors")
        
        # Shuffle the professors list to randomize selection
        random.shuffle(all_professors)
        
        # Take only the first max_professors professors
        professors_to_scrape = all_professors[:max_professors]
        
        for professor in professors_to_scrape:
            if professors_count >= max_professors:
                break
                
            # Skip if we've already scraped this professor
            if professor['id'] in self.scraped_professor_ids:
                logger.info(f"Skipping {professor['name']} (already scraped)")
                continue
                
            if not professor['url']:
                logger.info(f"Skipping {professor['name']} (no URL)")
                continue
            
            logger.info(f"Scraping reviews for {professor['name']}")
            
            professor_reviews = self.get_professor_reviews(professor['url'], professor['id'])
            
            # Limit the number of reviews per professor
            professor_reviews = professor_reviews[:max_reviews_per_professor]
            
            for review in professor_reviews:
                all_reviews.append({
                    'professor_name': professor['name'],
                    'department': professor['department'],
                    'professor_id': professor['id'],
                    'text': review['text'],
                    'rating': review['rating'],
                    'difficulty': review.get('difficulty', 0.0),
                    'would_take_again': review.get('would_take_again', None),
                    'class': review['class'],
                    'date': review['date'],
                    'tags': ', '.join(review['tags']) if isinstance(review['tags'], list) else review['tags'],
                    'professor_top_tags': ', '.join(review['professor_top_tags']) if isinstance(review['professor_top_tags'], list) else review['professor_top_tags']
                })
            
            # Add professor to scraped list
            self.scraped_professor_ids.add(professor['id'])
            
            professors_count += 1
            logger.info(f"Collected {len(professor_reviews)} reviews for {professor['name']}")
            
            # Add a delay between professor requests to avoid rate limiting
            time.sleep(random.uniform(1, 3))
        
        # Create DataFrame and save to CSV
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join('data', f'penn_state_reviews_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved {len(df)} reviews to {csv_path}")
            return csv_path
        else:
            logger.warning("No reviews were collected")
            return None

def main():
    scraper = RateMyProfessorsScraper()
    scraper.scrape_penn_state_reviews(max_professors=50, max_reviews_per_professor=20)


if __name__ == "__main__":
    main()