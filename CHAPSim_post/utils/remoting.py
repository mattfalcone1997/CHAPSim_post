import subprocess
import os

class RemoteSSH:
    def __init__(self,address):
        user_host = address.split('@')
        if len(user_host) != 2:
            msg = "Issue with address"
            raise ValueError(msg)
        
        self._address = address
        
            
    def get_file(self,remote_file,local_file=None):
        
        remote_cmd = self._address + ':' + remote_file 
        if local_file is None:
            local_file = os.path.basename(remote_file)
        
        cmd = ['scp',remote_cmd,local_file]
        
        out = subprocess.run(cmd)
        
        if out.returncode != 0:
            msg = f"Secure copy existed with error code {out.returncode}: {out.stderr}"
            raise Exception(msg)
        


        
        
        

    