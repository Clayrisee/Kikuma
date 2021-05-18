from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from predict_xception import predict
import numpy as np
import tempfile
import json

app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')

class Image(Resource):

    def post(self):
        args = parser.parse_args()
        the_file = args['file']
        ofile, ofname = tempfile.mkstemp()
        the_file.save(ofname)
        results = predict(ofname)
        labels = {0:'Acne', 1:'Chickenpox', 2:'Melanoma', 3:'Scabies'}
        label = np.argmax(results, axis=1)[0]
        output = {'probabilitas': results.tolist(), 'label': labels[label]}

        return json.dumps(output)


api.add_resource(Image, '/image')

if __name__ == '__main__':
    app.run(debug=True)