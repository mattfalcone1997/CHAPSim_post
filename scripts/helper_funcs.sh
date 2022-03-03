function test_return {
    if [ $? -ne 0 ]; then
        echo -e "Non-zero return code. $1. exiting"
        exit 1
    fi
}

function test_cmd {
    if ! hash $1 > /dev/null 2>/dev/null ; then 
        if [ $# -ge 2 ]; then
            echo -e "$2: exiting"
        else
            echo -e "Command $1 not found: exiting"
        fi
        exit 1
    fi
}

function chk_dir {
    stat $1 > /dev/null 2>/dev/null
    test_return "Folder $1 does not exist"
}