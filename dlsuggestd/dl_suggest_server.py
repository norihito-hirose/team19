""" This script DLSuggestServer with daemonize super class """
import sys
import os
import time
import signal
import atexit
import logging
from http.server import HTTPServer
from dl_suggest_handler import DLSuggestHandler

class Daemon(object):
    """
    A generic daemon class.
    Usage: subclass the Daemon class and override the run() method
    """

    def __init__(self, pidfile, stdin='/dev/null',
                 stdout='/dev/null', stderr='/dev/null'):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.pidfile = pidfile

    def daemonize(self):
        """
        do the UNIX double-fork magic, see Stevens' "Advanced
        Programming in the UNIX Environment" for details (ISBN 0201563177)
        http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
        """
        # Do first fork
        self.fork()

        # Decouple from parent environment
        self.dettach_env()

        # Do second fork
        self.fork()

        # Flush standart file descriptors
        sys.stdout.flush()
        sys.stderr.flush()

        #
        self.attach_stream('stdin', mode='r')
        self.attach_stream('stdout', mode='a+')
        self.attach_stream('stderr', mode='a+')

        # write pidfile
        self.create_pidfile()

    def attach_stream(self, name, mode):
        """
        Replaces the stream with new one
        """
        stream = open(getattr(self, name), mode)
        os.dup2(stream.fileno(), getattr(sys, name).fileno())

    def dettach_env(self): # pylint: disable=no-self-use
        """ dettach env"""
        os.chdir("/")
        os.setsid()
        os.umask(0)

    def fork(self): # pylint: disable=no-self-use
        """
        Spawn the child process
        """
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as excep:
            sys.stderr.write("Fork failed: %d (%s)\n" % (excep.errno, excep.strerror))
            sys.exit(1)

    def create_pidfile(self):
        """ create pidfile"""
        atexit.register(self.delpid)
        pid = str(os.getpid())
        logging.info("create pidfile %s" % self.pidfile)
        open(self.pidfile, 'w+').write("%s\n" % pid)

    def delpid(self):
        """
        Removes the pidfile on process exit
        """
        os.remove(self.pidfile)

    def start(self):
        """
        Start the daemon
        """
        # Check for a pidfile to see if the daemon already runs
        pid = self.get_pid()

        if pid:
            message = "pidfile %s already exist. Daemon already running?\n"
            sys.stderr.write(message % self.pidfile)
            sys.exit(1)

        # Start the daemon
        logging.info("daemonize")
        self.daemonize()
        logging.info("run")
        self.run()

    def get_pid(self):
        """
        Returns the PID from pidfile
        """
        try:
            pf_value = open(self.pidfile, 'r')
            pid = int(pf_value.read().strip())
            pf_value.close()
        except (IOError, TypeError):
            pid = None
        return pid

    def stop(self, silent=False):
        """
        Stop the daemon
        """
        # Get the pid from the pidfile
        pid = self.get_pid()

        if not pid:
            if not silent:
                message = "pidfile %s does not exist. Daemon not running?\n"
                sys.stderr.write(message % self.pidfile)
            return # not an error in a restart

        # Try killing the daemon process
        try:
            while True:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
        except OSError as err:
            err = str(err)
            if err.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                sys.stdout.write(str(err))
                sys.exit(1)

    def restart(self):
        """
        Restart the daemon
        """
        self.stop(silent=True)
        self.start()

    def run(self):
        """
        You should override this method when you subclass Daemon.
        It will be called after the process has been
        daemonized by start() or restart().
        """
        raise NotImplementedError

class DLSuggestServer(Daemon):
    """
    Simple http server
    """

    def run(self):
        DLSuggestHandler.model = self.model
        server = HTTPServer(('127.0.0.1', self.port), DLSuggestHandler)
        logging.info("starting server... listen port: %s" % self.port)
        server.serve_forever()

    # pylint: disable=attribute-defined-outside-init
    def set_port(self, port):
        """ set server listen port """
        self.port = port

    def set_model(self, model):
        """ set eval request """
        self.model = model
