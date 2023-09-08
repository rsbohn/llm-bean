"extra beans provided by llm"
import click
import llm
#from llm import get_embedding_model
import sqlite_utils
from datetime import datetime

@llm.hookimpl
def register_commands(cli):
    @cli.command(name="bean")
    @click.argument("content")
    @click.option("-m", "--model", help="Embedding model to use")
    def bean(content:str, model:str):
        """
        Write one of your deep thoughts.

        \b
            llm bean "My cornfield is full of ears."
        
        Saves your thought in the 'beans' collection.

        \b
            llm similar beans -c "full of beans"
        
        Prints beans similar to the provided context.
        """
        collection = "beans"
        db = sqlite_utils.Database(llm.user_dir() / "embeddings.db")
        collection_obj = None
        model_obj = None
        if llm.Collection.exists(db, collection):
            collection_obj = llm.Collection(collection, db)
            model_obj = collection_obj.model()
        else:
            if not model:
                ## use the user selected default model
                model = "sentence-transformers/all-MiniLM-L6-v2"
                if model is None:
                    raise click.ClickException("I have no model!")
            collection_obj = llm.Collection(collection, db=db, model_id=model)
            model_obj = collection_obj.model

        if model_obj is None:
            try:
                model_obj = llm.get_embedding_model(model)
            except llm.UnknownModelError:
                raise click.ClickException("No model provided. :sad_panda:")

        # use `now` for the entry id
        now = datetime.now()
        timestamp_id = now.strftime("%Y.%m.%dT%H%M")
        collection_obj.embed(timestamp_id, content, store=True)
        print(timestamp_id, content[:48])