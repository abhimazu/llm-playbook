import json
import os
import requests
from openai import OpenAI
from justwatch import JustWatch


# -------------------------
# Configuration / API Keys
# -------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("Please set your OMDB_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

# -------------------------
# Memory Module
# -------------------------
class Memory:
    """
    This class manages a JSON-based memory that stores each user's preferences.
    The preferences include:
      - movies: a list of movie titles the user likes.
      - genres: a list of genres the user likes.
      - country: the user's country (used for streaming availability checks).
    """
    def __init__(self, memory_file="user_memory.json"):
        self.memory_file = memory_file
        self.memory = {}
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)
        else:
            self.memory = {}

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def get_user_preferences(self, user_id):
        if user_id not in self.memory:
            # Initialize with empty/default values.
            self.memory[user_id] = {"movies": [], "genres": [], "country": None}
            self.save_memory()
        return self.memory[user_id]

    def update_user_preferences(self, user_id, movies=None, genres=None, country=None):
        prefs = self.get_user_preferences(user_id)
        if movies:
            if isinstance(movies, list):
                for movie in movies:
                    if movie not in prefs["movies"]:
                        prefs["movies"].append(movie)
            elif isinstance(movies, str):
                movie_list = [m.strip() for m in movies.split(",") if m.strip()]
                for movie in movie_list:
                    if movie not in prefs["movies"]:
                        prefs["movies"].append(movie)
        if genres:
            if isinstance(genres, list):
                for genre in genres:
                    if genre not in prefs["genres"]:
                        prefs["genres"].append(genre)
            elif isinstance(genres, str):
                genre_list = [g.strip() for g in genres.split(",") if g.strip()]
                for genre in genre_list:
                    if genre not in prefs["genres"]:
                        prefs["genres"].append(genre)
        if country:
            prefs["country"] = country
        self.memory[user_id] = prefs
        self.save_memory()
        return prefs

# -------------------------
# API Helper Functions
# -------------------------
def get_movie_details(movie_title):
    """
    Fetch movie details using the OMDb API.
    """
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={movie_title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("Response") == "True":
            return data
    return None

def check_movie_availability(movie_title, country):
    """
    Check if a movie is available in a given country using the JustWatch API.
    """
    justwatch = JustWatch(country=country)
    results = justwatch.search_for_item(query=movie_title)
    if results:
        return results[0]  # Return the first matching result.
    return None

# -------------------------
# Agent Class
# -------------------------
class MovieRecommenderAgent:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = Memory()

    def set_movies(self, movies):
        """
        Set/update the user's favorite movies. 'movies' is a list.
        """
        prefs = self.memory.update_user_preferences(self.user_id, movies=movies)
        return f"Updated movies preferences: {prefs['movies']}."

    def set_genres(self, genres):
        """
        Set/update the user's favorite genres. 'genres' is a list.
        """
        prefs = self.memory.update_user_preferences(self.user_id, genres=genres)
        return f"Updated genres preferences: {prefs['genres']}."

    def set_country(self, country):
        """
        Set/update the user's country.
        """
        prefs = self.memory.update_user_preferences(self.user_id, country=country)
        return f"Updated country preference: {prefs['country']}."

    def get_recommendations(self):
        """
        Generate movie recommendations based on the stored preferences.
        The LLM is prompted with the current preferences (movies, genres, country)
        and asked to recommend 5 new movies. Then, each recommendation is validated
        for availability in the user's country via JustWatch.
        """
        prefs = self.memory.get_user_preferences(self.user_id)
        movies = prefs.get("movies", [])
        genres = prefs.get("genres", [])
        country = prefs.get("country")
        missing = []
        if not movies:
            missing.append("movies")
        if not genres:
            missing.append("genres")
        if not country:
            missing.append("country")
        if missing:
            return f"Your preferences are incomplete. Please set: {', '.join(missing)}."

        prompt = (
            f"My current movie preferences are: Movies: {movies}, Genres: {genres}, Country: {country}. "
            "Based on these, recommend 5 new movies that I might like and that are available in my country. "
            "Return the movie titles as a JSON array."
        )
        messages = [
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            recommendation_text = response.choices[0].message.content.strip()
            try:
                # Expecting a JSON array from the LLM.
                recommended_movies = json.loads(recommendation_text)
                if not isinstance(recommended_movies, list):
                    recommended_movies = [str(recommended_movies)]
            except Exception:
                # Fallback: split by commas.
                recommended_movies = [m.strip() for m in recommendation_text.split(",") if m.strip()]

            # Validate each recommended movie for availability.
            available_movies = []
            for movie in recommended_movies:
                avail = check_movie_availability(movie, country)
                if avail:
                    available_movies.append(movie)
            if available_movies:
                result = "Based on your preferences, here are recommended movies available in your country:\n" + "\n".join(available_movies)
            else:
                result = "None of the recommended movies appear to be available in your country."
            return result
        except Exception as e:
            return f"Error generating recommendations: {e}"

    def check_availability(self, movies, country=None):
        """
        Check availability of a list of movies in a given country.
        If 'country' is not provided, the stored country is used.
        """
        if country is None:
            prefs = self.memory.get_user_preferences(self.user_id)
            country = prefs.get("country")
            if not country:
                return "Country not set in your preferences."
        results = {}
        for movie in movies:
            avail = check_movie_availability(movie, country)
            results[movie] = avail if avail else "Not available"
        text = "Availability:\n"
        for movie, avail in results.items():
            if avail != "Not available":
                text += f"{movie}: Available\n"
            else:
                text += f"{movie}: Not available\n"
        return text

# -------------------------
# Function Definitions for LLM Function Calling
# -------------------------
function_definitions = [
    {
        "name": "set_movies",
        "description": "Set the list of movies the user likes.",
        "parameters": {
            "type": "object",
            "properties": {
                "movies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of movies the user likes."
                }
            },
            "required": ["movies"]
        }
    },
    {
        "name": "set_genres",
        "description": "Set the list of genres the user likes.",
        "parameters": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of genres the user likes."
                }
            },
            "required": ["genres"]
        }
    },
    {
        "name": "set_country",
        "description": "Set the user's country for movie availability.",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Country code or name where the user is located."
                }
            },
            "required": ["country"]
        }
    },
    {
        "name": "get_recommendations",
        "description": "Get movie recommendations based on the user's stored preferences.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "check_availability",
        "description": "Check the availability of a list of movies in a given country.",
        "parameters": {
            "type": "object",
            "properties": {
                "movies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of movies to check availability for."
                },
                "country": {
                    "type": "string",
                    "description": "Country code or name to check availability in."
                }
            },
            "required": ["movies", "country"]
        }
    }
]

# -------------------------
# Dispatcher and LLM Function Calling
# -------------------------
def run_agent(agent, user_input):
    """
    This function sends the user input along with the current user preferences
    (movies, genres, country) to the LLM. The LLM decides (using function calling)
    which function to invoke. The code dispatches the call to the corresponding method,
    then sends the function result back to the LLM for a final user-friendly response.
    """
    # Load current preferences and include them in the system prompt.
    prefs = agent.memory.get_user_preferences(agent.user_id)
    system_context = (
        f"Current user preferences: Movies: {prefs.get('movies', [])}, "
        f"Genres: {prefs.get('genres', [])}, Country: {prefs.get('country', 'not set')}. "
        "If any of these are missing, please ask the user to provide them. "
        "Use the provided functions accordingly."
    )
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_input}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # A model that supports function calling.
        messages=messages,
        functions=function_definitions,
        function_call="auto"
    )
    
    message = response.choices[0].message
    
    # If the LLM decided to call a function, dispatch accordingly.
    if message.function_call:
        function_name = message.function_call.name
        arguments = message.function_call.get("arguments")
        try:
            args = json.loads(arguments) if arguments else {}
        except Exception as e:
            args = {}
        
        if function_name == "set_movies":
            movies = args.get("movies")
            function_response = agent.set_movies(movies)
        elif function_name == "set_genres":
            genres = args.get("genres")
            function_response = agent.set_genres(genres)
        elif function_name == "set_country":
            country = args.get("country")
            function_response = agent.set_country(country)
        elif function_name == "get_recommendations":
            function_response = agent.get_recommendations()
        elif function_name == "check_availability":
            movies = args.get("movies")
            country = args.get("country")
            function_response = agent.check_availability(movies, country)
        else:
            function_response = f"Function {function_name} is not implemented."
        
        # Append the function call and its output to the conversation.
        messages.append(message)
        messages.append({
            "role": "function",
            "name": function_name,
            "content": function_response
        })
        
        # Ask the LLM to generate a final response.
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        final_message = second_response.choices[0].message.content
        print("Assistant:", final_message)
    else:
        # If no function call was made, simply output the assistant's message.
        print("Assistant:", message.content)

# -------------------------
# Main Interactive Loop
# -------------------------
def main():
    user_id = "user123"  # Replace with dynamic user ID as needed.
    agent = MovieRecommenderAgent(user_id)
    
    print("Welcome to the Enhanced Movie Recommender Agent!")
    print("You can update your preferences (movies, genres, country) or ask for recommendations.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYour query: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        run_agent(agent, user_input)

if __name__ == "__main__":
    main()