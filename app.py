from flask import Flask
from flask_restful import Api, Resource

#It creates a Flask web application and initializes a Flask-RESTful API.
app = Flask(__name__)
api = Api(app)

#predictive data  analysis
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

recipe_df = pd.read_csv(r'C:\personal nidhi\FULLSTACK\recipe\Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
recipe_df['Ingredients'] = recipe_df['Ingredients'].str.lower()


#The CountVectorizer converts the list of ingredients for each recipe into a matrix of token counts. The cosine similarity matrix is then computed based on these ingredient counts.
vectorizer = CountVectorizer()
ingredient_matrix = vectorizer.fit_transform(recipe_df['Ingredients'].apply(lambda x: ', '.join(eval(x))))#convert list to string
similarity_matrix = cosine_similarity(ingredient_matrix, ingredient_matrix)#close to 1 if similar ingredients

#It has a get method that takes an ingredient_name as a parameter and returns a JSON response containing recipe recommendations.
class RecommendationResource(Resource):
    def get(self, ingredient_name):
        return {'recommendations': recommend_by_ingredients(ingredient_name)}

def recommend_by_ingredients(ingredient_name):
    ingredient_index = recipe_df['Ingredients'].apply(lambda x: ingredient_name.lower() in x).idxmax()#checks in ingredient column if present,index of the first occurrence of the maximum value. 
    similarity_scores = similarity_matrix[ingredient_index]#retrieves the row of similarity matrix of the required ingredient and shows similarity score
    similar_recipes_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[1:6]#selecting the indices of the top 5 most similar recipes to the specified ingredient based on their cosine similarity scores. 
    recommended_titles = [recipe_df.iloc[index]['Title'] for index in similar_recipes_indices]#extract titles
    return recommended_titles

# Add the RecommendationResource to the API with the '/recommend/<ingredient_name>' route
api.add_resource(RecommendationResource, '/recommend/<ingredient_name>')


#run the flask app
if __name__ == "__main__":
    app.run(debug=True)

