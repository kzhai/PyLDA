#!/usr/bin/python
#
#-------------------------------------------------------------------------------
# Filename: peer.py
# Version: 0.2
# Description:
#     Define a class Peer, having all data structures needed for a peer.
#     250_again_and_again
#-------------------------------------------------------------------------------

class Peer:
    pId = -1
    routeTable = None
    leafNodes = None
    neighbors = None

    def __init__(self, pId, comm):
        self.pId = pId
        print "A peer with pId: " + str(pId) + " is created!"

    def getPId(self):
        return self.pId


    def join(self, contactPeer):
        if contactPeer == None:
            print "The contact Peer has not been specified."
            return
        print str(self.pId) + " is joining peer " + str(self.contactPeer.getPId())

    def terminate(self):
        print str(self.pId) + " is terminating ..."

    def leave(self):
        print str(self.pId) + " has left the system."

    def stablize(self):
        print "stablize"

    def route(self, key):
        print "Routing key " + key + " from peer " + str(self.pId)

    def localOperation(self):
        print "*** Welcome to peer " + str(self.pId) + '! ***'
        while (True):
            print '''Please select operation:
            \t'j') join the contacted peer;
            \t'l') volunteerly leave the network;
            \t't') terminate without telling anyone;
            \t'r') search a key
            \t'q') logout peer.'''
                
            option = raw_input("Your command: ")

            if option == 'j':
                self.join(None)
            elif option == 'l':
                self.leave()
                return
            elif option == 't':
                self.terminate()
                return
            if option == 'q':
                return
            else:
                print "Please choose an option:"
                continue

class Communicator:
    def __init__(self, peer_list, pid_list):
        print "A communicator is created!"
