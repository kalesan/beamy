import sys
import os
import glob
import urllib
import pandas as pd

from http.server import HTTPServer, SimpleHTTPRequestHandler

HOST_NAME="0.0.0.0"
PORT = 8008

class PythonServer(SimpleHTTPRequestHandler):
    """Python HTTP Server that handles GET and POST requests"""
    def do_GET(self):
        if self.path == '/':
            dirs = [os.path.basename(os.path.normpath(d)) for d in glob.glob('./sims/*/')]
            self.send_response(200, "OK")
            self.end_headers()
            self.wfile.write(bytes('\n'.join(['simulation'] + dirs), encoding='utf-8'))
        else:
            path = os.path.split(self.path)

            if path[-1].startswith('iteration'):
                path = os.path.join('./sims', './'+path[0], 'iteration.csv')
                df = pd.read_csv(path)

                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

                ret = df

                if 'info' in query.keys():
                    if query['info'] == 'BS':
                        ret = df['BS'].unique()
                    elif query['info'] == 'UE':
                        ret = df['UE'].unique()
                    elif query['info'] == 'Nr':
                        ret = df['Nr'].unique()
                    elif query['info'] == 'Nt':
                        ret = df['Nt'].unique()
                    elif query['info'] == 'SNR':
                        ret = df['SNR'].unique()

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
            elif path[-1] == 'info':
                path = os.path.join('./sims', './'+path[0], 'info.csv')
                df = pd.read_csv(path)
                self.send_response(200, "OK")
                self.end_headers()
                self.wfile.write(bytes(df.to_csv(), 'utf-8'))
            else:
                self.send_response_only(400, "Not found")

if __name__ == "__main__":
    server = HTTPServer((HOST_NAME, PORT), PythonServer)

    print(f"Server started http://{HOST_NAME}:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("Server stopped succesfully")
        sys.exit(0)