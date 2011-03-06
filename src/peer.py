#!/usr/bin/python
#
#-------------------------------------------------------------------------------
# Filename: peer.py
# Version: 0.3
# Description:
#     Define a class Peer, having all data structures needed for a peer.
# Update:
#     For 0.3: send/receive is added to each peer.
#     use peer.send(objective_Peer_Id, message) to send a message;
#     use peer.rcv() to rcv all messages into local messageList.
#     received messages will not be received again by calling rcv();
#     local messageList will be overritten by multiple call of rcv().
#-------------------------------------------------------------------------------

class Peer:
    pId = -1
    routeTable = None
    leafNodes = None
    neighbors = None
    comm = None # Communicator
    messageList = []

    def __init__(self, pId, comm):
        self.pId = pId
        self.comm = comm
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

    def send(self, objPId, message):
        self.comm.send(self.pId, objPId, message)

    def rcv(self):
        self.messageList = self.comm.rcv(self.pId)

    def printMessage(self):
        for item in self.messageList:
            print '"' + str(item[2]) + '" from peer ' + str(item[0]) + '.'
    def prtMsgQueue(self):
        self.comm.prtMsgQueue()

    def localOperation(self):
        print "*** Welcome to peer " + str(self.pId) + '! ***'
        while (True):
            print '''Please select operation:
            \t'j') join the contacted peer;
            \t'l') volunteerly leave the network;
            \t't') terminate without telling anyone;
            \t'send') send a message;
            \t'rcv') receive all messages;
            \t'prt') print all messages;
            \t'allMsg') print all flying messages;
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
            elif option == 'send':
                objId = int(raw_input("Please input the object pId: "))
                message = raw_input("Please input the message: ")
                self.send(objId, message)
                print "Message has been sent."
            elif option == 'rcv':
                self.rcv()
                print str(len(self.messageList)) + " new message(s)!"
            elif option == 'prt':
                self.printMessage()
            elif option == 'allMsg':
                self.prtMsgQueue()
            elif option == 'q':
                return
            else:
                print "Please choose an option:"
                continue

class Communicator:
    messageQueue = []    
    def __init__(self, peer_list, pid_list):
        print "A communicator is created!"

    def send(self, sourceId, destId, content):
        tuple = (sourceId, destId, content)
        self.messageQueue.append(tuple)

    def rcv(self, myPId):
        msgList = []
        newMsgQueue = []
        for item in self.messageQueue:
            if item[1] == myPId:
                msgList.append(item)
            else:
                newMsgQueue.append(item)
        self.messageQueue = newMsgQueue
        return msgList

    def prtMsgQueue(self):
        for item in self.messageQueue:
            print item
