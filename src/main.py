#!/usr/bin/python
#
#-------------------------------------------------------------------------------
# Filename: main.py
# Version: 0.3
# Description:
#     A simulated client giving commands to the P2P system.
# Update:
#     0.2: Add communication mechanism
#     0.3: Use sequential ID temporarily for peers
#-------------------------------------------------------------------------------

import random

from peer import *

# client's local list that keeps all peer's ID.
pid_list = []
# list of peer objects, having the same idx with pid_list.
peer_list = []
# communicator
comm = Communicator(peer_list, pid_list)

# start shell

print "Starting the system ..."

while(True):
    print '''Please select:
    \t'l') list existing peers;
    \t'a') add a new peer;
    \t'g') goto a peer for more operations
    \t'r') route;
    \t'q') quite.'''
    option = raw_input("Your command: ")

    # List all peer pIds.
    if option == 'l': 
        print "\tThere is/are " + str(len(pid_list)) + " peer(s) in the system."
        for item in pid_list:
            print '\t' + str(item)

    # add a new peer with random yet unique pId
    elif option == 'a':
        # pid = random.randint(0,1024)
        pid = len(pid_list) # for simplicity, use sequential id first.
        if pid not in pid_list: # to make sure the pid is unique
            pid_list.append(pid)
            tmpPeer = Peer(pid, comm) 

            peer_list.append(tmpPeer) # add peer to the list
            
            
        print "peer " + str(pid) + " added."


    #goto a peer for more operations.
    if option == 'g':
        if len(peer_list) == 0:
            print "No peer exists yet."
            continue
        while (True):
            inputtedId = raw_input("Please input the pId you want to login:")
            objPeerId = int(inputtedId)
            if int(objPeerId) not in pid_list:
                print "Please choose an existing peer ID."
                continue
            break
        objPeer = peer_list[pid_list.index(objPeerId)]
        objPeer.localOperation()


    #Terminate the whole system by simply exit.
    elif option == 'q':
        if len(peer_list) != 0:
            suboption = raw_input("Peers still running. Are you sure to \
quite?(yes/no)")
            if suboption == "yes":
                print "Terminating system ..."
                exit(0)
        else:
            exit(0)
    else:
        print "Please select a command from the list."
