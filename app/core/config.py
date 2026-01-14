from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the settings class for your application.
# This class will automatically load environment variables from a .env file.
# The `model_config` attribute with `SettingsConfigDict` is used to specify
# the source of the environment variables. In this case, it's a `.env` file.
class Settings(BaseSettings):
    """
    Represents the application settings.

    This class inherits from Pydantic's BaseSettings, which allows it to
    automatically load environment variables. The variables are defined as
    attributes of this class.
    """
    GROQ_API_KEY: str
    COHERE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX: str

    # `SettingsConfigDict` is used to configure the behavior of the settings class.
    # `env_file=".env"` tells Pydantic to load environment variables from a file named `.env`.
    # `env_file_encoding='utf-8'` specifies the encoding of the .env file.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

# Create an instance of the Settings class.
# This instance will be imported by other modules to access the settings.
settings = Settings()

# --- How to use this config in other files ---
#
# To use these settings in other parts of your application, you can import
# the `settings` object.
#
# For example, in `app/core_services/llm_clients/groq_llm.py`, you could use it like this:
#
# from app.core.config import settings
# from groq import Groq
#
# client = Groq(api_key=settings.GROQ_API_KEY)
#
# # Now you can use the `client` object to make API calls.
#
#
# Similarly, for Cohere:
#
# from app.core.config import settings
# import cohere
#
# co = cohere.Client(api_key=settings.COHERE_API_KEY)
#
#
# And for Pinecone:
#
# from app.core.config import settings
# import pinecone
#
# pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)
# index = pinecone.Index(settings.PINECONE_INDEX)