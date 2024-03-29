import yara
import os
import time
import json
import logging
import hashlib
import atexit
from configparser import ConfigParser

import tflearn
from numpy import argmax
from flask import Flask, request, redirect, render_template, url_for, abort, jsonify

import training


config = ConfigParser()
config.read('config.ini')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config['api']['upload_path']
app.config['MAX_CONTENT_LENGTH'] = int(config['api']['upload_max_length'])

logging.basicConfig(
    level=logging.DEBUG, filename='demo.log', filemode='w',
    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'
)


class TempFile:

    def __init__(self, path, name):
        self.path = os.path.abspath(path)
        self.name = name

    def get_name(self):
        return self.name

    def get_path(self):
        return os.path.realpath(os.path.join(self.path, self.name))

    def __del__(self):
        # file = os.join(self.path, self.name)
        # if os.path.isfile(file):
        #     os.remove(self.file)
        pass


def vaild_file(filename):
    ALLOWED_EXTENSIONS = ['php']
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def yarcat():
    if os.path.exists("./rules/output.yara") == True:
        os.remove("./rules/output.yara")
    with open("./rules/output.yara", "wb") as outfile:
        for root, dirs, files in os.walk("./rules", topdown=False):
            for name in files:
                fname = str(os.path.join(root, name))
                with open(fname, "rb") as infile:
                    if fname != './rules/output.txt':
                        outfile.write(infile.read())


def compileandscan(filematch):
    yarcat()

    rules = yara.compile('./rules/output.yara')
    matches = rules.match(filematch, timeout=60)
    ma = 0
    length = len(matches)
    if length > 0:
        c = matches
        dmatch = []
        for match in matches:
            dmatch.append(matches[ma].strings)
            ma = ma + 1
    else:
        matches = 'No YARA hits.'
        dmatch = None
    return [matches, dmatch]


def check_with_model(file_id):
    global model
    file = TempFile(os.path.join(app.config['UPLOAD_FOLDER']), file_id)
    ###
    file_opcodes = [training.get_file_opcode(file.get_path())]
    training.serialize_codes(file_opcodes)
    file_opcodes = tflearn.data_utils.pad_sequences(file_opcodes, maxlen=seq_length, value=0.)

    res_raw = model.predict(file_opcodes)
    res = {
        # revert from categorical
        'judge': True if argmax(res_raw, axis=1)[0] else False,
        'chance': float(res_raw[0][argmax(res_raw, axis=1)[0]]),
    }
    return res


@app.route('/check/result/<file_id>')
def check_webshell(file_id):
    if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], file_id)):
        a = compileandscan('./uploads/' + file_id)
        ur = {'filename': file_id, 'yararesults': a[0], 'yarastrings': a[1]}

        logging.info('checking file: {0}'.format(file_id))
        res_check = check_with_model(file_id)
        res = {
            'file_id': file_id,
            'malicious_judge': res_check['judge'],
            'malicious_chance': res_check['chance'],
        }

    return render_template('result.html', ur=ur, json_result=json.dumps(res))


@app.route('/check/upload', methods=['GET', 'POST'])
def receive_file():
    if request.method == 'POST':
        file = request.files['file[]']
        if file and vaild_file(file.filename):
            file_id = hashlib.md5((file.filename+str(time.time())).encode('utf-8')).hexdigest()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_id))
            return redirect(url_for('check_webshell', file_id=file_id))
        else:
            return abort(400)

    elif request.method == 'GET':
        return render_template('upload.html')


@app.route('/')
def index():
    return redirect(url_for('receive_file'))


@atexit.register
def atexit():
    logging.info('detection stopped')


if __name__ == '__main__':
    global model, seq_length

    host = config['server']['host']
    port = int(config['server']['port'])
    model_record = config['training']['model_record']
    seq_length = json.load(open(model_record, 'r'))['seq_length']

    logging.info('loading model...')
    model = training.get_model()

    logging.info('detection started')
    app.run(host='0.0.0.0', port=port, debug=True)
