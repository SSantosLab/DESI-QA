
import Pyro4 as pyro
import sys
import os

@pyro.expose
class Math(object):
    def sum(self,a,b):
        print('sum {} + {} '.format(a,b))
        return int(a)+int(b)

def main():
    server = Math()
    daemon = pyro.Daemon()
    ns = pyro.locateNS(host=os.environ['NS_HOST'],
                       port=int(os.environ['NS_PORT']))
    uri = daemon.register(server)
    ns.register('Math',uri)
    print('Ready object : ',uri)
    daemon.requestLoop()
if __name__ == '__main__':
    if 'NS_HOST' not in os.environ or 'NS_PORT' not in os.environ:
        print('You have to set VARS NS_HOST and NS_PORT')
        sys.exit(1)
    else:
        print("STARTING")
    main()
