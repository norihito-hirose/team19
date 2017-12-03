""" This script DLSuggentServers response handler """
import re
import json
import logging
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from dl_suggest_model import DLSuggestModel

TOP_N = 10

class DLSuggestHandler(BaseHTTPRequestHandler):
    """
    Simple http handler
    """
    model = None

    # pylint: disable=invalid-name
    def do_GET(self):
        """ GET request """
        if re.search('/v1/predict*', self.path) != None:
            logging.debug(self.path)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # get query parameter in the url path
            query = urlparse(self.path).query
            if query:
                # parse query param string to dict
                input_code = parse_qs(query)
                if 'in' in input_code:
                    try:
                        # predict
                        pred = self.model.predict(input_code['in'], TOP_N)
                    except AttributeError:
                        logging.error('not defined predict method on specified %s',
                                      self.model.__class__)
                        err = self._render_errors(
                            'LoadModelError',
                            'not defined predict method on specified %s' % type(self.model)
                        )
                        self.wfile.write(err)
                        return

                    pred['info'] = self.model.get_info(self.path)

                    try:
                        res = json.dumps(pred).encode(encoding='utf-8')
                    except TypeError as e:
                        err = self._render_errors(
                            'JSONParseError',
                            str(e)
                        )
                        self.wfile.write(err)
                        return

                    self.wfile.write(res)
                else:
                    logging.warning("not exsists 'in' param")
                    err = self._render_errors(
                        'InvalidURLParam',
                        "need 'in=' param request url. %s" % self.path
                    )
                    self.wfile.write(err)
            else:
                logging.warning("not exists query params")
                err = self._render_errors(
                    'InvalidURLParam',
                    "need 'in=' param request url. %s" % self.path
                )
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            logging.warning("404 not found")
            err = self._render_errors(
                'NotFound',
                'Not found %s' % self.path
            )
            self.wfile.write(err)
        return

    def _render_errors(self, err_type, desc):
        return json.dumps({
            'info': self.model.get_info(self.path),
            'errors': [
                {
                    'type': err_type,
                    'desc': desc
                }
            ]
        }).encode(encoding='utf-8')
