import sys
import os
import glob
import urllib
import pandas as pd

from http.server import HTTPServer, SimpleHTTPRequestHandler

SIMS_PATH = os.environ.get("SIMS_PATH", "./sims")

HOST_NAME = "0.0.0.0"
PORT = 8008


class PythonServer(SimpleHTTPRequestHandler):
    """Python HTTP Server that handles GET and POST requests"""
    def do_GET(self):
        if self.path == '/':
            dirs = [os.path.basename(os.path.normpath(d))
                    for d in glob.glob('./sims/*/')]

            self.send_response(200, "OK")
            self.end_headers()
            self.wfile.write(bytes('\n'.join(['simulation'] + dirs),
                             encoding='utf-8'))
        else:
            path = os.path.split(self.path)

            if path[-1].startswith('iteration'):
                self.process_iteration(path)
            elif path[-1] == 'info':
                self.process_info(path)
            else:
                self.send_response_only(400, "Not found")

    def process_iteration(self, path):
        path = os.path.join(SIMS_PATH, './'+path[0], 'iteration.csv')
        df = pd.read_csv(path)

        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        ret = df

        if 'info' in query.keys() and query['info'] in df.columns:
            return df[query['info']].unique()

        if 'Nr' in query.keys():
            ret = ret[ret['Nr'] == int(query['Nr'][0])]
        if 'Nt' in query.keys():
            ret = ret[ret['Nt'] == int(query['Nt'][0])]
        if 'BS' in query.keys():
            ret = ret[ret['BS'] == int(query['BS'][0])]
        if 'UE' in query.keys():
            ret = ret[ret['UE'] == int(query['UE'][0])]
        if 'SNR' in query.keys():
            ret = ret[ret['SNR'] == int(query['SNR'][0])]

        self.send_response(200, "OK")
        self.end_headers()
        self.wfile.write(bytes(ret.to_csv(), 'utf-8'))

    def process_info(self, path):
        path = os.path.join(SIMS_PATH, './'+path[0], 'info.csv')
        df = pd.read_csv(path)
        self.send_response(200, "OK")
        self.end_headers()
        self.wfile.write(bytes(df.to_csv(), 'utf-8'))


if __name__ == "__main__":
    server = HTTPServer((HOST_NAME, PORT), PythonServer)

    print(f"Server started http://{HOST_NAME}:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("Server stopped succesfully")
        sys.exit(0)
