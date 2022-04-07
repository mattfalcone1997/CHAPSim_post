from imp import new_module
from paramiko import SSHClient, rsakey, pkey, ssh_exception
import getpass
from scp import SCPClient, SCPException
import os

class RemoteSSH:
    def __init__(self,address):
        user_host = address.split('@')
        if len(user_host) != 2:
            msg = "Issue with address"
            raise ValueError(msg)
        
        self._hostname = user_host[-1]
        self._user = user_host[0]
        self.__password = None    
        self.__keyfile = None
        
    def set_keyfile(self,keyfile):
        if not os.path.isfile(keyfile):
            msg = f"Keyfile specified {keyfile}, not found"
            raise FileNotFoundError(msg)
        
        
        self.__pkey_password = self._get_pkey_pword(keyfile)           
        self.__keyfile = keyfile
        
    def _get_pkey_pword(self,keyfile,password=None):
        try:
            rsakey.RSAKey.from_private_key_file(open(keyfile,'r'),password=password)
            return password
        
        except ssh_exception.PasswordRequiredException:
            msg = f"Passphrase for key file {keyfile}"
            password = getpass.getpass(msg)
            return  self._get_pkey_pword(keyfile,password=password)
        
    def _get_password(self):
        if self.__password is None:
            name = "@".join([self._user,self._hostname])
            self.__password = getpass.getpass("Password for %s:"%name)
            
    def get_file(self,file_name,name=None):
        self._get_password()
        
        client = SSHClient()
        client.load_system_host_keys()
        
        try:
            client.connect(self._hostname,
                        username=self._user,
                        password=self.__password,
                        key_filename=self.__keyfile,
                        passphrase=self.__pkey_password)
        except ssh_exception.AuthenticationException as e:
            msg = "SSh client raised an exception. You may need to set private key file:\n"
            raise ssh_exception.AuthenticationException(msg+e.args[0]) from None
        
        if name is None:
            name = os.path.basename(file_name)
            
        scp = SCPClient(client.get_transport())
        try:
            scp.get(file_name,name)
        except SCPException:
            msg = f"SCP client raised an exception: {e.args[0]}"
            raise SCPException from None
        
        
        
        

    