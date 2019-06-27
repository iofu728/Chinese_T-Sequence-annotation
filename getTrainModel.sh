#!/bin/bash
# @Author: gunjianpan
# @Date:   2019-06-27 13:55:53
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-27 14:54:29

SIGN_1='#-#-#-#-#-#-#-#-#-#'
SIGN_2='---__---'
SIGN_3='**************'
INS='Instaling'
DOW='Downloading'
ERROR_MSG='Sorry,Pip_not_found'

# echo color
RED='\033[1;91m'
GREEN='\033[1;92m'
YELLOW='\033[1;93m'
BLUE='\033[1;94m'
CYAN='\033[1;96m'
NC='\033[0m'

echo_color() {
    case ${1} in
    red) echo -e "${RED} ${2} ${NC}" ;;
    green) echo -e "${GREEN} ${2} ${NC}" ;;
    yellow) echo -e "${YELLOW} ${2} ${NC}" ;;
    blue) echo -e "${BLUE} ${2} ${NC}" ;;
    cyan) echo -e "${CYAN} ${2} ${NC}" ;;
    *) echo ${2} ;;
    esac
}

if [ ! -z "$(which pip3 2>/dev/null | sed -n '/\/pip3/p')" ]; then
    PIP=pip3
elif [ ! -z "$(which pip 2>/dev/null | sed -n '/\/pip/p')" ]; then
    PIP=pip
else
    echo_color red ${ERROR_MSG} && exit 1
fi

echo $PIP

check_install() {
    if [ -z "$(which ${1} 2>/dev/null | sed -n '/\/'${1}'/p')" ]; then
        echo_color green "${SIGN_1} ${INS} ${1} ${SIGN_1}"
        $PIP install gdown --user
    fi

}

check_install gdown
gdown https://drive.google.com/uc\?id\=1npxA_pcEvIa4c42rho1HgnfJ7tamThSy
