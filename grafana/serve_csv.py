import sys
import os
import glob

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

            if path[-1] == 'iteration':
                path = os.path.join('./sims', './'+path[0], 'iteration.csv')
                with open(path) as f:
                    self.send_response(200, "OK")
                    self.end_headers()
                    self.wfile.write(bytes(f.read(), 'utf-8'))
            elif path[-1] == 'info':
                path = os.path.join('./sims', './'+path[0], 'info.csv')
                with open(path) as f:
                    self.send_response(200, "OK")
                    self.end_headers()
                    self.wfile.write(bytes(f.read(), 'utf-8'))
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