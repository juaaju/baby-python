from flask import Flask, jsonify, request
from flask_cors import CORS 
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from sam_satu_kamera import all_params

def base64_to_opencv_image(base64_string):
    # Remove the prefix 'data:image/jpeg;base64,' or similar
    base64_data = base64_string.split(',')[1]

    # Decode the Base64 string to bytes
    image_data = base64.b64decode(base64_data)

    # Convert the bytes to a NumPy array
    np_arr = np.frombuffer(image_data, np.uint8)

    # Decode the NumPy array to an OpenCV image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# Contoh data sementara
data = [
    {
        "id": "recent",
        "image1":"",
    }
]

@app.route('/api/items', methods=['GET'])
def get_items():
    
    result = all_params("image1.jpg")
    return jsonify({"items": data,"result":result})

# Route untuk mendapatkan data berdasarkan ID
@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((item for item in data if item["id"] == item_id), None)
    if item:
        return jsonify({"item": item})
    else:
        return jsonify({"message": "Item not found"}), 404

# Route untuk menambahkan data baru
@app.route('/api/items', methods=['POST'])
def add_item():
    new_item = request.get_json()
    data.append(new_item)
    return jsonify({"message": "Item added successfully", "item": new_item}), 201

# Route untuk mengupdate data berdasarkan ID
@app.route('/api/items/<string:item_id>', methods=['PUT'])
def update_item(item_id):
    item = next((item for item in data if item["id"] == item_id), None)
    if item:
        item.update(request.get_json())
        base64_string1 = data[0]["image1"]
        
        string1=base64_string1[base64_string1.index(",")+1:]
        
        base64_decode1 = base64.b64decode(string1)
        

        img1 = Image.open(BytesIO(base64_decode1))
        
        img1 = img1.convert("RGB")
        
        img1.save("image1.jpg")
        
        return jsonify({"message": "Item updated successfully", "item": item})
    else:
        return jsonify({"message": "Item not found"}), 404

# Route untuk menghapus data berdasarkan ID
@app.route('/api/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global data
    data = [item for item in data if item["id"] != item_id]
    return jsonify({"message": "Item deleted successfully"})

if __name__ == '__main__':
    app.run(debug=True)
