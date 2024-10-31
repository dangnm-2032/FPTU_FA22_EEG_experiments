import PySimpleGUI as sg
################## CONFIG ###################
config_empty={'connection_controller_status':None,'user_status':None,'connection_muse_status':None}
config={'connection_controller_status':None,'user_status':None,'connection_muse_status':None}

################## check controller connection #################
def check_controller_connection():
    return 'Connected'
menu_def = ['Port 1:','Port 2:','Port 3:','Port 4:','Port 5:','Port 6:','Port 7:','Port 8:','Port 9:']
lst = sg.Listbox(menu_def, size=(10, 5), font=('Arial Bold', 14), enable_events=True, key='-LIST-')
################# check muse connection ###################
def check_muse_connection():
    return 'Connected'
################# check user ready ################
def check_user_ready():
    if config['user_status']=='Ready':
        return 'Ready'
################ data preprocessing and feed to model ########
def get_data(data):
    pass
def preprocess(data):
    pass
def model(data):
    pass
################ Model output ###############
commands =[
    sg.Text('Left', size=(15, 5), justification='c',key='L', text_color='white', relief=sg.RELIEF_SUNKEN),
    sg.Text('Right', size=(15, 5), justification='c', key='R', relief=sg.RELIEF_SUNKEN),
    sg.Text('Both', size=(15, 5), justification='c', key='B', relief=sg.RELIEF_SUNKEN),
    sg.Text('Clenching', size=(15, 5), justification='c', key='C', relief=sg.RELIEF_SUNKEN),
    sg.Text('Eyebrows', size=(15, 5), justification='c', key='E', relief=sg.RELIEF_SUNKEN)
]
def update_command(command):
    if command == 'L':
        window['L'].update(text_color='red')
    if command == 'R':
        window['R'].update(text_color='red')
    if command == 'B':
        window['B'].update(text_color='red')
    if command == 'C':
        window['C'].update(text_color='red')
    if command == 'E':
        window['E'].update(text_color='red')
################ MAIN ######################

layout = [
    # layout controller connection
    [sg.Text('Connection Controller status (OK, NOT OK)', size=(15, 5), justification='left', relief=sg.RELIEF_SUNKEN)],
    [sg.Text(size=(10, 1), font=('Helvetica', 14), key='-CONNECTION_STATUS_CTL-', relief=sg.RELIEF_SUNKEN)],
    # layout muse connection
    [sg.Text('Connection Muse status (OK, NOT OK)', size=(15, 5), justification='left', relief=sg.RELIEF_SUNKEN)],
    [sg.Text(size=(10, 1), font=('Helvetica', 14), key='-CONNECTION_STATUS_MUSE-', relief=sg.RELIEF_SUNKEN)],
    # layout user status
    [sg.Text('User status (Ready, NOT Ready)', size=(15, 5), justification='center', relief=sg.RELIEF_SUNKEN)],
    [sg.Text(size=(10, 1), font=('Helvetica', 14), key='-USERS_STATUS-', justification='center', relief=sg.RELIEF_SUNKEN)],
    [sg.B('Ready',key='-button_ready-',button_color=("white", "blue"))],
    # layout available port for controller
    [lst],
    [commands],
    # OK, Refresh, Exit
    [sg.Column([[sg.OK(), sg.B('Refresh'), sg.Exit()]], element_justification='right', expand_x=True)]
]
keys = ['-button_ready-','-USERS_STATUS-','-CONNECTION_STATUS-','-LIST-']
# Create the window
window = sg.Window("Robot Control Interface", layout, size=(1000, 800), finalize=True)

# draw_initial()

# Event loop
c = 0
while True:
    event, values = window.read()
    print(event,values)
    if event == sg.WIN_CLOSED or event=='Exit':
        break
    # check connection controller connection status from config
    if config['connection_controller_status'] == None:
        window['-CONNECTION_STATUS_CTL-'].update(value='Not connected')
    # check connection muse connection status from config
    if config['connection_muse_status'] == None:
        window['-CONNECTION_STATUS_MUSE-'].update(value='Not connected')
    # check user status
    if config['user_status'] == None:
        window['-USERS_STATUS-'].update(value='Not ready')
    # when user hit ready, event = -button_ready-, save to config file
    if event == '-button_ready-':
        config['user_status']='Ready'
        window['-button_ready-'].update(button_color=("white", "green"))
    if event=='OK':
        ##### CHECK CONNECTION CONTROLLER STATUS #####
        PORT_VALUE = values
        print(PORT_VALUE['-LIST-'])
        result_connection_controller = check_controller_connection()
        config['connection_controller_status'] = result_connection_controller
        window['-CONNECTION_STATUS_CTL-'].update(value=result_connection_controller)
        ##### CHECK CONNECTION MUSE STATUS #####
        result_connection_muse = check_muse_connection()
        config['connection_muse_status'] = result_connection_muse
        window['-CONNECTION_STATUS_MUSE-'].update(value=result_connection_muse)
        ##### CHECK USER STATUS #####
        result_user = check_user_ready()
        config['user_status'] = result_user
        window['-USERS_STATUS-'].update(value=result_user)
    if config['user_status'] == 'Ready' and config['connection_controller_status'] == 'Connected' and config['connection_muse_status'] == 'Connected':
        data=''
        ######## pull data from muse #######
        get_data(data)
        ######## preprocessing ##########
        preprocess(data)
        ######## feed to model ##########
        model(data)
        command = 'L'
        update_command(command=command)
    if event=='Refesh':
        config=config_empty
        for key in keys:
            window[key]('')
    print('count: ',c)
    c+=1

# Close the window
window.close()
