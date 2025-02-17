# main.py
import json
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from justwatch import JustWatch
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
os.environ["OMDB_API_KEY"] = ""

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
    Manages a JSON-based memory that stores each user's preferences:
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
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.memory = data
                else:
                    print("Warning: Memory file format is invalid. Resetting memory.")
                    self.memory = {}
            except Exception as e:
                print(f"Error loading memory: {e}. Resetting memory.")
                self.memory = {}
        else:
            self.memory = {}

    def save_memory(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=4)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def get_user_preferences(self, user_id: str):
        if user_id not in self.memory:
            # Initialize with empty/default values.
            self.memory[user_id] = {"movies": [], "genres": [], "country": None}
            self.save_memory()
        return self.memory[user_id]

    def update_user_preferences(
        self, user_id: str, movies=None, genres=None, country=None
    ):
        prefs = self.get_user_preferences(user_id)
        if movies:
            # Accept both list and comma-separated string formats.
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
def get_movie_details(movie_title: str):
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


def get_country_code(country: str) -> str:
    """
    Map full country names to ISO country codes.
    If the country is not found in the mapping, return the input unchanged.
    """
    mapping = {
        "canada": "CA",
        "united states": "US",
        "usa": "US",
        "uk": "GB",
        "united kingdom": "GB",
        # Add more mappings as needed.
    }
    return mapping.get(country.lower(), country)


def check_movie_availability(movie_title: str, country: str):
    """
    Check if a movie is available in a given country using the JustWatch API.
    Converts a full country name to its ISO code if necessary.
    Returns the first matching result if available, otherwise None.
    """
    country_code = get_country_code(country)
    try:
        justwatch = JustWatch(country=country_code)
        results = justwatch.search_for_item(query=movie_title)
        if results and isinstance(results, list) and len(results) > 0:
            # Optionally, log the results for debugging.
            return results[0]
    except Exception as e:
        print(f"Error in JustWatch API for '{movie_title}' in '{country_code}': {e}")
    return None


def get_similar_movies(user_preferences, num_recommendations=5):
    """
    Uses OpenAI’s ChatCompletion API to generate movie recommendations.
    The prompt asks for movies similar to those the user likes.
    """
    prompt = (
        f"I like the following movies: {', '.join(user_preferences.get('movies', []))}. "
        f"My preferred genres are: {', '.join(user_preferences.get('genres', []))}. "
        f"Based on these, please recommend {num_recommendations} similar movies that are available in my country. "
        "Return only the movie titles, each on a separate line."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful movie recommendation assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=150
        )
        recommendation_text = response.choices[0].message.content.strip()
        recommendations = [
            line.strip(" -") for line in recommendation_text.split("\n") if line.strip()
        ]
        # Fallback: if the model returned a comma-separated list.
        if len(recommendations) < num_recommendations:
            recommendations = [
                rec.strip() for rec in recommendation_text.split(",") if rec.strip()
            ]
        return recommendations[:num_recommendations]
    except Exception as e:
        print(f"Error in get_similar_movies: {e}")
        return []


# -------------------------
# Movie Recommender Agent
# -------------------------
class MovieRecommenderAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = Memory()

    def set_movies(self, movies):
        prefs = self.memory.update_user_preferences(self.user_id, movies=movies)
        return f"Updated movies preferences: {prefs['movies']}."

    def set_genres(self, genres):
        prefs = self.memory.update_user_preferences(self.user_id, genres=genres)
        return f"Updated genres preferences: {prefs['genres']}."

    def set_country(self, country):
        prefs = self.memory.update_user_preferences(self.user_id, country=country)
        return f"Updated country preference: {prefs['country']}."

    def get_recommendations(self):
        """
        Generate movie recommendations based on stored preferences.
        If any preference is missing, ask the user to provide it.
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
            # Instead of an error, ask the user for the missing details.
            return f"It seems I don't have your {', '.join(missing)} preference{'s' if len(missing) > 1 else ''}. Could you please provide {'them' if len(missing) > 1 else 'it'}?"

        # If all preferences are available, generate recommendations.
        prompt = (
            f"My current movie preferences are: Movies: {movies}, Genres: {genres}, Country: {country}. "
            "Based on these, recommend 5 new movies that I might like and that are available in my country. "
            "Return the movie titles as a JSON array."
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful movie recommendation assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=150
            )
            recommendation_text = response.choices[0].message.content.strip()
            try:
                recommended_movies = json.loads(recommendation_text)
                if not isinstance(recommended_movies, list):
                    recommended_movies = [str(recommended_movies)]
            except Exception:
                recommended_movies = [
                    m.strip() for m in recommendation_text.split(",") if m.strip()
                ]

            available_movies = []
            for movie in recommended_movies:
                if check_movie_availability(movie, country):
                    available_movies.append(movie)
            if available_movies:
                return (
                    "Based on your preferences, here are recommended movies available in your country:\n"
                    + "\n".join(available_movies)
                )
            else:
                return "None of the recommended movies appear to be available in your country."
        except Exception as e:
            return f"Error generating recommendations: {e}"

    def check_availability(self, movies, country=None):
        """
        Check availability of a list of movies in a given country.
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
                    "description": "List of movies the user likes.",
                }
            },
            "required": ["movies"],
        },
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
                    "description": "List of genres the user likes.",
                }
            },
            "required": ["genres"],
        },
    },
    {
        "name": "set_country",
        "description": "Set the user's country for movie availability.",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Country code or name where the user is located.",
                }
            },
            "required": ["country"],
        },
    },
    {
        "name": "get_recommendations",
        "description": "Get movie recommendations based on the user's stored preferences.",
        "parameters": {"type": "object", "properties": {}},
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
                    "description": "List of movies to check availability for.",
                },
                "country": {
                    "type": "string",
                    "description": "Country code or name to check availability in.",
                },
            },
            "required": ["movies", "country"],
        },
    },
]


# -------------------------
# Pydantic Models for API
# -------------------------
class UserPreferencesModel(BaseModel):
    movies: list[str]
    genres: list[str]
    country: str | None


class ChatRequest(BaseModel):
    message: str
    preferences: UserPreferencesModel


class ChatResponse(BaseModel):
    reply: str
    updatedMemory: UserPreferencesModel | None = None


# -------------------------
# FastAPI Application Setup
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# API Endpoints
# -------------------------
@app.get("/api/memory", response_model=UserPreferencesModel)
async def get_memory():
    """Return the current user preferences for user 'user123'."""
    agent = MovieRecommenderAgent("user123")
    return agent.memory.get_user_preferences("user123")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Accept a user chat message along with current preferences.
    Use OpenAI function calling to decide on actions (such as updating memory,
    getting recommendations, or asking for missing info) and return the final reply and updated memory.
    """
    agent = MovieRecommenderAgent("user123")
    # Update stored memory with provided preferences.
    agent.memory.update_user_preferences(
        "user123",
        movies=request.preferences.movies,
        genres=request.preferences.genres,
        country=request.preferences.country,
    )

    prefs = agent.memory.get_user_preferences("user123")
    system_context = (
        f"Current user preferences: Movies: {prefs.get('movies', [])}, "
        f"Genres: {prefs.get('genres', [])}, Country: {prefs.get('country', 'not set')}. "
        "If any of these are missing, please ask the user to provide them using the appropriate functions."
    )
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": request.message},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=function_definitions,
            function_call="auto",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {e}")

    message = response.choices[0].message

    if message.function_call:
        function_name = message.function_call.name
        arguments = message.function_call.arguments
        try:
            args = json.loads(arguments) if arguments else {}
        except Exception:
            args = {}

        if function_name == "set_movies":
            function_response = agent.set_movies(args.get("movies"))
        elif function_name == "set_genres":
            function_response = agent.set_genres(args.get("genres"))
        elif function_name == "set_country":
            function_response = agent.set_country(args.get("country"))
        elif function_name == "get_recommendations":
            function_response = agent.get_recommendations()
        elif function_name == "check_availability":
            function_response = agent.check_availability(
                args.get("movies"), args.get("country")
            )
        else:
            function_response = f"Function {function_name} is not implemented."

        messages.append(
            {
                "role": message.role,
                "content": message.content,
                "function_call": {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                },
            }
        )
        messages.append(
            {"role": "function", "name": function_name, "content": function_response}
        )

        try:
            second_response = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error calling OpenAI API (second call): {e}"
            )
        final_message = second_response.choices[0].message.content
        reply = final_message
    else:
        reply = message.content

    updated_memory = agent.memory.get_user_preferences("user123")
    return ChatResponse(reply=reply, updatedMemory=updated_memory)


# -------------------------
# Run with: uvicorn main:app --reload
# -------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
