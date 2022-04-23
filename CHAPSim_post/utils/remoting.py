import subprocess
import os

class RemoteSSH:
    def __init__(self,address):
        user_host = address.split('@')
        if len(user_host) != 2:
            msg = "Issue with address"
            raise ValueError(msg)
        
        self._address = address
        
    def _user_ready(self):            
        x = input(f"Are you ready to log in to ssh at {self._address} (y or n)?")
        if x.lower() == 'n':
            msg = "You are not ready"
            raise RuntimeError(msg)
        elif x.lower() == 'y':
            pass
        else:
            print("You have given an invalid input.")
            self._user_ready()
                    
    def get_file(self,remote_file,local_file=None):
        
        self._user_ready()
        
        remote_cmd = self._address + ':' + remote_file 
        if local_file is None:
            local_file = os.path.basename(remote_file)
        
        cmd = ['scp',remote_cmd,local_file]
        
        out = subprocess.run(cmd)
        
        if out.returncode != 0:
            msg = f"Secure copy existed with error code {out.returncode}: {out.stderr}"
            raise Exception(msg)
        


        
        
        

    