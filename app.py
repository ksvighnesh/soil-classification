from flask import Flask, request, render_template, send_from_directory
from PIL import Image
from PIL import UnidentifiedImageError
from tensorflow import keras
import numpy as np
import io
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

crop_map = {
    'Black': ['Cotton: A soft, fluffy fiber that is grown in warm climates and used to make textiles. It is a major cash crop in many countries, including the United States, China, and India.',
              'Tobacco: A plant that is grown for its leaves, which are dried and used to make cigarettes, cigars, and other tobacco products. It is a controversial crop due to its addictive and harmful properties.',
              'Sorghum: A grain crop that is widely cultivated in Africa and Asia. It is used for a variety of purposes, including human consumption, animal feed, and biofuel production. It is a hardy crop that can tolerate drought and other adverse conditions.'],
    
    'Red': ['Wheat: A cereal grain that is grown all over the world and is a staple food in many cultures. It is high in protein and carbohydrates, making it a good source of energy and nutrition.', 
            'Pulses: A type of legume that includes beans, lentils, and peas. They are a good source of protein and other nutrients, and are commonly used in many cuisines around the world.', 
            'Groundnut: Also known as peanuts, this is a legume crop that is grown for its edible seeds. It is high in protein and healthy fats, and is a common ingredient in many foods, such as peanut butter and snack foods.'],
    
    'Alluvial': ['rice: A cereal grain that is the staple food for over half of the world\'s population. It is high in carbohydrates and is a good source of energy. Rice is grown in many countries, particularly in Asia, and there are many different varieties.', 
                'sugarcane: A tall, perennial grass that is grown for its sweet juice, which is used to make sugar and other sweeteners. Sugarcane is a major cash crop in many countries, particularly in tropical regions.', 
                'maize: Also known as corn, this is a cereal grain that is widely grown all over the world. It is a staple food in many cultures, and is used for a variety of other purposes, such as animal feed and biofuel production.'],

    'Desert': ['Cactus: A type of succulent plant that is adapted to hot, dry climates. Some varieties are edible and are used in traditional Mexican cuisine.', 
                'Date palm: A type of palm tree that is grown for its sweet fruit, the date. Dates are a popular food in many cultures and are also used to make syrup and other sweeteners.', 
                'Jojoba: A shrub that is native to the southwestern United States and Mexico. It produces a nut that is used to make jojoba oil, which is a popular ingredient in cosmetics and other personal care products.']
}

class_labels = ['Alluvial', 'Black', 'Desert', 'Red']

model = keras.models.load_model('soil_type_classification_model_97.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        
        try:
            try:
                image_data = file.read()
                if image_data is None:
                    return render_template('result.html', output="Enter valid image")
                
                img = Image.open(io.BytesIO(image_data))
                output = processing_function(img)
                crops = crop_map[output]
                percentages = get_prediction_percentages(img)
                print(output, percentages)
                if output in class_labels and percentages[output] >= 70:
                    crops = crop_map[output]
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    img.save(filepath)
                    return render_template('result.html', soil_type=output, crops=crops, percentages=percentages,class_labels=class_labels, file=file)
                else:
                    return render_template('error.html', output="Unable to classify the image. Please try with another image.")
            
            except UnidentifiedImageError as e:
                msg = f"{e}. Please upload a valid image file."
                return render_template('error.html', output=msg)
        except Exception as e:
            return render_template('error.html', output=e)
    else:
        return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def processing_function(image):
    img = image.resize((1024, 1024))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_label = np.argmax(prediction)
    class_label_string = class_labels[class_label]
    return class_label_string

def get_prediction_percentages(img):
    percentages = np.zeros(len(class_labels))
    img = img.resize((1024, 1024))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]

    for i, p in enumerate(prediction):
        percentages[i] = round(p * 100, 2)
    return dict(zip(class_labels, percentages))

if __name__ == '__main__':
    app.run(debug=True)
