import PySimpleGUI as sg
import serial
import threading
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from FPTU_FA24_EEG_Artifacts_Recognition.conponents.backend import *
# from test10 import *
config_manager = ConfigurationManager()
app = ControlBackend(config_manager)
################## FAKE COMMAND #############
command=['L','R','B','C','E']*100
################## CONFIG ###################
config={'connection_controller_status':None,'user_status':None,'connection_muse_status':None,'has_predict':None}
################## check controller connection #################
def check_controller_connection(port_value):
    app.controller_port = port_value
    app.connect_controller()
    if app.status_controller_connect:
        return 'Connected'
    else:
        return 'Not connected'
menu_def = app.list_all_ports()
# menu_def = ['Port 1:','Port 2:','Port 3:','Port 4:','Port 5:','Port 6:','Port 7:','Port 8:','Port 9:']
lst = sg.Listbox(menu_def, size=(10, 5), font=('Arial Bold', 14), enable_events=True, key='-PORT_LIST-')
################# check muse connection ###################
def check_muse_connection():
    app.connect_muse_stream()
    if app.status_muse_connect:
        return 'Connected'
    else:
        return 'Not connected'
elapsed_time = 0
before_status=None
start = time.time()
def check_muselsl_stream():
    global elapsed_time, before_status, start
    while True:
        elapsed_time = time.time() - start
        # print(elapsed_time)
        if app.status_muse_stream != before_status:
            start = time.time()
            before_status = app.status_muse_stream
        if app.status_muse_stream and elapsed_time>16:
            current_color = "green"
            text = 'Muse Found'
            window["blink_box"].update(background_color=current_color)
            window["blink_box"].update(text)
        else:
            current_color = "red"
            text = 'Searching for Muse'
            window["blink_box"].update(background_color=current_color)
            time.sleep(1.5)
            window["blink_box"].update(background_color='blue')
            window["blink_box"].update(text)
        time.sleep(0.1)
        # window.write_event_value('blinking', None)
        window.refresh()
################# check user ready ################
def check_user_ready():
    if config['user_status']=='Ready':
        return 'Ready'

############## update command ################
def update_command(command):
    if config['has_predict'] != None:
        dct[config['has_predict']].update(button_color=("black", "lightgrey"))
    dct[command].update(button_color=("black", "red"))
    config['has_predict']=command
############## draw #################
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)
    return figure_canvas_agg
def create_plot():
    plot_duration = 2 # seconds
    SR = 256
    x = np.linspace(0, plot_duration, SR*plot_duration)

    fig = plt.figure(figsize=(12, 4),dpi=100)

    temp = int(SR * plot_duration / 2)
    max = 1
    min = 0

    line_TP9_raw, = plt.plot(x, [0, 1,]  * temp, alpha=0.6)
    line_AF7_raw, = plt.plot(x, [0, 1,]  * temp, alpha=0.6)
    line_AF8_raw, = plt.plot(x, [0, 1,]  * temp, alpha=0.6)
    line_TP10_raw, = plt.plot(x, [0, 1,]  * temp, alpha=0.6)

    line_TP9_filter, = plt.plot(x, [0, 1,]  * temp)
    line_AF7_filter, = plt.plot(x, [0, 1,]  * temp)
    line_AF8_filter, = plt.plot(x, [0, 1,]  * temp)
    line_TP10_filter, = plt.plot(x, [0, 1,]  * temp)


    line_eyebrows, = plt.plot(x, [0, 1,] * temp, label='eyebrows')
    line_left, = plt.plot(x, [0, 1,] * temp, label='left')
    line_right, = plt.plot(x, [0, 1,] * temp, label='right')
    line_both, = plt.plot(x, [0, 1,] * temp, label='both')
    line_teeth, = plt.plot(x, [0, 1,] * temp, label='teeth')
    plt.ylim(-2, 12.5)
    plt.legend(loc=1)
    plt.axis('off')
    plt.text(-0.2, -0.1, 'PREDICT')
    plt.text(-0.2, 10.8, 'TP9')
    plt.text(-0.2, 8.8, 'AF7')
    plt.text(-0.2, 6.8, 'AF8')
    plt.text(-0.2, 4.8, 'TP10')
    buffer = np.zeros((SR*plot_duration, 13))
    dct = {
        'max':max,'min':min,
        'line_TP9_raw':line_TP9_raw,'line_AF7_raw':line_AF7_raw,'line_AF8_raw':line_AF8_raw,'line_TP10_raw':line_TP10_raw,
        'line_TP9_filter':line_TP9_filter,'line_AF7_filter':line_AF7_filter,'line_AF8_filter':line_AF8_filter,'line_TP10_filter':line_TP10_filter,
        'line_eyebrows':line_eyebrows,'line_left':line_left,'line_right':line_right,'line_both':line_both,'line_teeth':line_teeth,
        'buffer':buffer
    }
    return fig,dct
def draw(dct,eeg_data,input,y_pred):
    buffer = dct['buffer']
    line_TP9_raw=dct['line_TP9_raw']
    line_AF7_raw=dct['line_AF7_raw']
    line_AF8_raw=dct['line_AF8_raw']
    line_TP10_raw=dct['line_TP10_raw']
    line_TP9_filter=dct['line_TP9_filter']
    line_AF7_filter=dct['line_AF7_filter']
    line_AF8_filter=dct['line_AF8_filter']
    line_TP10_filter=dct['line_TP10_filter']
    line_eyebrows=dct['line_eyebrows']
    line_left=dct['line_left']
    line_right=dct['line_right']
    line_both=dct['line_both']
    line_teeth=dct['line_teeth']
    n_timesteps = 64
    buffer[:-n_timesteps] = buffer[n_timesteps:]

    buffer[-n_timesteps:, 0] = eeg_data[:, 0] /200 + 11 # TP9
    buffer[-n_timesteps:, 1] = input[0, 4, :, 0] * 0.2 + 11 # TP9 filter

    buffer[-n_timesteps:, 2] = eeg_data[:, 1] /200 + 9 # AF7
    buffer[-n_timesteps:, 3] = input[0, 5, :, 0] * 0.2 + 9 # AF7 filter

    buffer[-n_timesteps:, 4] = eeg_data[:, 2] /200 + 7 # AF8
    buffer[-n_timesteps:, 5] = input[0, 6, :, 0] * 0.2 + 7 # AF8 filter

    buffer[-n_timesteps:, 6] = eeg_data[:, 3] /200 + 5 # TP10
    buffer[-n_timesteps:, 7] = input[0, 7, :, 0] * 0.2 + 5 # TP10 filter

    buffer[-n_timesteps:, 8] = y_pred[0, :, 1] + 0
    buffer[-n_timesteps:, 9] = y_pred[0, :, 2] + 0
    buffer[-n_timesteps:, 10] = y_pred[0, :, 3] + 0
    buffer[-n_timesteps:, 11] = y_pred[0, :, 4] + 0
    buffer[-n_timesteps:, 12] = y_pred[0, :, 5] + 0
    
    line_TP9_raw.set_ydata(buffer[:, 0])
    line_TP9_filter.set_ydata(buffer[:, 1])

    line_AF7_raw.set_ydata(buffer[:, 2])
    line_AF7_filter.set_ydata(buffer[:, 3])

    line_AF8_raw.set_ydata(buffer[:, 4])
    line_AF8_filter.set_ydata(buffer[:, 5])

    line_TP10_raw.set_ydata(buffer[:, 6])
    line_TP10_filter.set_ydata(buffer[:, 7])

    line_eyebrows.set_ydata(buffer[:, 8])
    line_left.set_ydata(buffer[:, 9])
    line_right.set_ydata(buffer[:, 10])
    line_both.set_ydata(buffer[:, 11])
    line_teeth.set_ydata(buffer[:, 12])

    fig.canvas.draw()
    fig.canvas.flush_events()
##### define layout #####
controller = [
    [sg.Text('Connection Controller status',font=('Helvetica', 20), size=(15, 5), justification='left', relief=sg.RELIEF_SUNKEN)],
    [sg.Text(size=(10, 1), font=('Helvetica', 14), key='-CONNECTION_STATUS_CTL-', relief=sg.RELIEF_SUNKEN, expand_x=True)],
    [sg.B('Connect',key='-button_controller-',button_color=("white", "blue")),sg.B('Disconnect',key='-button_controller_disconnect-',button_color=("white", "blue"))]
]
muse_connection = [
    [sg.Text('Connection Muse status',font=('Helvetica', 20), size=(15, 5), justification='center', relief=sg.RELIEF_SUNKEN)],
    [sg.Text(size=(12, 1), font=('Helvetica', 14), key='-CONNECTION_STATUS_MUSE-',justification='center', relief=sg.RELIEF_SUNKEN)],
    [sg.B('Connect',key='-button_muse-',button_color=("white", "blue")),sg.B('Disconnect',key='-button_muse_disconnect-',button_color=("white", "blue"))]
]
user_status = [
    [sg.Text('User status',font=('Helvetica', 20), size=(15, 5), justification='center', relief=sg.RELIEF_SUNKEN)],
    [sg.Text(size=(10, 1), font=('Helvetica', 14), key='-USERS_STATUS-', justification='center', relief=sg.RELIEF_SUNKEN)],
    [sg.B('Ready',key='-button_ready-',button_color=("white", "blue"))]
]

commands = [
    [sg.Text('', size=(10, 1)), sg.Button("FORWARD", key='C', size=(10, 3),button_color=("black", "lightgrey")), sg.Text('', size=(10, 1))],
    [sg.Button("LEFT", key='L', size=(10, 3),button_color=("black", "lightgrey")), sg.Button("STOP", key='B', size=(10, 3),button_color=("black", "lightgrey")), sg.Button("RIGHT", key='R', size=(10, 3),button_color=("black", "lightgrey"))],
    [sg.Text('', size=(10, 1)), sg.Button("BACKWARD", key='E', size=(10, 3),button_color=("black", "lightgrey")), sg.Text('', size=(10, 1))]
]

# blinking_box = sg.Text("Searching for Muse stream ", background_color="red", key="blink_box",font=("Helvetica", 15))
blinking_box = sg.Frame(
        "",  # No title for the outer frame
        [
            [sg.Text("Notification", justification='center', expand_x=True,font=("Helvetica", 15),size=(50, 1))],  # Label in the large box
            [sg.Text("Searching for Muse", background_color="red", key="blink_box",font=("Helvetica", 10),size=(20, 1))],
            [sg.Text("", background_color="lightblue", key="notification",font=("Helvetica", 10),size=(30, 5),auto_size_text=True)]  # Small text box inside large box
        ],
        size=(300, 150),  # Size of the large box
        background_color="lightblue"
    )
lst = sg.Listbox(menu_def, size=(10, 5), font=('Arial Bold', 14), enable_events=True, key='-PORT_LIST-')
layout = [
    # layout controller connection, layout muse connection
    [sg.Column(controller), sg.Column(muse_connection, element_justification='center'), sg.Column(user_status),blinking_box],
    # layout available port for controller
    [sg.Text('AVAILABLE PORT', size=(15, 5), key=('-CHOOSE_PORT-'))],
    [lst,sg.Canvas(key="-CANVAS-")],
    [sg.Column(commands, element_justification='center', justification='center', expand_x=True),sg.Button("NOTHING",key='None', size=(10, 3))],
    [sg.Column([[sg.B('INFERFENCE', key='-INFERENCE-',button_color=("white", "blue")),sg.B('STOP INFERFENCE',key='-NOT_INFERENCE-',button_color=("white", "blue"))]], element_justification='right', expand_x=True)],
    [sg.Column([[sg.B('Refresh'), sg.Exit()]], element_justification='right', expand_x=1,expand_y=1)]
]
####### Manual control robot #############
def manual_control(command):
    try:
        if command in ('LRBEC'):
            update_command(command=command)
            if command == 'E':
                print('Eyebrows - Go backward')
                app.ser.write(bytes('2', 'ascii'))
            elif command == 'L':
                print('Left - Turn left')
                app.ser.write(bytes('5', 'ascii'))
            elif command == 'R':
                print('Right - Turn right')
                app.ser.write(bytes('6', 'ascii'))
            elif command == 'B':
                print('Both - Stop')
                app.ser.write(bytes('0', 'ascii'))
            elif command == 'C':
                print('Teeth - Go forward')
                app.ser.write(bytes('1', 'ascii'))
            else:
                return
    except:
        return 
# Create the window
window = sg.Window("Robot Control Interface", layout, location=(None, None), finalize=True)
####### define dictionary ########
dct = {
    'None':window['None'],
    'L':window['L'],
    'R':window['R'],
    'B':window['B'],
    'E':window['E'],
    'C':window['C']
}
dct_pred = {
    0:'None',
    1:'E',
    2:'L',
    3:'R',
    4:'B',
    5:'C'
}
def reset_config(config):
    for key in config:
        config[key]=None
# update port list
def update_port_list():
    menu_def = app.list_all_ports()
    window['-PORT_LIST-'].update(menu_def)
def refresh(config):
    global elapsed_time, before_status, start
    window['-CONNECTION_STATUS_CTL-'].update(value='Not connected')
    window['-CONNECTION_STATUS_MUSE-'].update(value='Not connected')
    window['-USERS_STATUS-'].update(value='Not Ready')
    window['-button_ready-'].update(button_color=("white", "blue"))
    window['-button_muse-'].update(button_color=("white", "blue"))
    window['-button_muse_disconnect-'].update(button_color=("white", "blue"))
    window['-button_controller-'].update(button_color=("white", "blue"))
    window['-INFERENCE-'].update(button_color=("white", "blue"))
    window['notification'].update('')
    window['notification'].update(background_color="lightblue")
    if config['has_predict'] is not None:
        dct[config['has_predict']].update(button_color=("black", "lightgrey"))
    elapsed_time = 0
    before_status=None
    start = time.time() 
    reset_config(config=config)
    update_port_list()
    app.exit()
refresh(config)
# define threads
def define_thread_1():
    t1 = threading.Thread(target=app.check_infinity, daemon=True)
    return t1
t1=define_thread_1()
t1.start()
def define_thread_2():
    blink = threading.Thread(target=check_muselsl_stream, daemon=True)
    return blink
### blinking
blink = define_thread_2()
blink.start()
def run_infer(dct,plot):
    while True:
        pred,eeg_data,input,y_pred = app.inference()
        if plot:
            draw(dct,eeg_data,input,y_pred)
        update_command(dct_pred[pred])
        window.write_event_value('-INFER-', pred)
        window.refresh()
def define_thread(dct,plot):
    t2 = threading.Thread(target=run_infer,args=(dct,plot), daemon=True)
    return t2
### draw initital
def ask_yes_no_question():
    # Show the Yes/No popup
    response = sg.PopupYesNo('Do you want to plot?')
    if response == 'Yes':
        plot=True
    elif response == 'No':
        plot=False
    else:
        print("Popup was closed")
    return plot
plot=ask_yes_no_question()
if plot:
    fig,dct_draw = create_plot()
    canvas_elem = window["-CANVAS-"]
    canvas = canvas_elem.TKCanvas
    fig_canvas_agg = draw_figure(canvas, fig)
else:
    dct_draw=None
####################### MAIN #######################
while True:
    window.refresh()
    if not blink.is_alive():
        blink = define_thread_2()
        blink.start()
    event, values = window.read()
    print(event,values)
    if event == sg.WIN_CLOSED or event=='Exit':
        app.exit()
        break
    manual_control(event)
    if event=='Refresh':
        refresh(config)
    # check user status
    if event == '-button_ready-':
        if config['user_status']!='Ready':
            window['-button_ready-'].update(button_color=("white", "green"))
            config['user_status']='Ready'
            window['-USERS_STATUS-'].update(value='Ready')
        elif config['user_status']=='Ready':
            window['-button_ready-'].update(button_color=("white", "blue"))
            config['user_status']=None
            window['-USERS_STATUS-'].update(value='Not Ready')
    if event == '-button_controller-':
        ##### CHECK CONNECTION CONTROLLER STATUS #####
        try:
            if app.status_controller_connect:
                app.ser=None
                app.status_controller_connect=False
            PORT_VALUE = values['-PORT_LIST-'][0]
            result_connection_controller = check_controller_connection(PORT_VALUE)
            if result_connection_controller == 'Connected':
                config['connection_controller_status'] = result_connection_controller
                window['-CONNECTION_STATUS_CTL-'].update(value=result_connection_controller)
                window['-button_controller-'].update(button_color=("white", "green"))
                window['-button_controller_disconnect-'].update(button_color=("white", "blue"))
            elif result_connection_controller == 'Not connected':
                config['connection_controller_status'] = None
                window['-CONNECTION_STATUS_CTL-'].update(value=result_connection_controller)
                window['-button_controller-'].update(button_color=("white", "red"))
                window['-button_controller_disconnect-'].update(button_color=("white", "blue"))
        except:
            window['-CONNECTION_STATUS_CTL-'].update(value='Choose port')
    if event == '-button_controller_disconnect-':
        try:
            app.ser.close()
            window['-button_controller_disconnect-'].update(button_color=("white", "red"))
            window['-button_controller-'].update(button_color=("white", "blue"))
            window['-CONNECTION_STATUS_CTL-'].update(value='Not connected')
        except:
            window['-button_controller_disconnect-'].update(button_color=("white", "red"))
            window['-button_controller-'].update(button_color=("white", "blue"))
            window['-CONNECTION_STATUS_CTL-'].update(value='Not connected')
    if event == '-button_muse-':
        ##### CHECK CONNECTION MUSE STATUS #####
        result_connection_muse = check_muse_connection()
        if result_connection_muse == 'Connected':
            config['connection_muse_status'] = result_connection_muse
            window['-CONNECTION_STATUS_MUSE-'].update(value=result_connection_muse)
            window['-button_muse-'].update(button_color=("white", "green"))
            window['-button_muse_disconnect-'].update(button_color=("white", "blue"))
        elif result_connection_muse == 'Not connected':
            config['connection_muse_status'] = None
            window['-CONNECTION_STATUS_MUSE-'].update(value=result_connection_muse)
            window['-button_muse-'].update(button_color=("white", "red"))
            window['-button_muse_disconnect-'].update(button_color=("white", "blue"))
    if event == '-button_muse_disconnect-':
        app.exit()
        window['-button_muse_disconnect-'].update(button_color=("white", "red"))
        window['-button_muse_disconnect-'].update('Disconnect')
        window['-button_muse-'].update(button_color=("white", "blue"))
    if event=='-INFERENCE-':
        ##### CHECK ALL STATUS #####
        if window['-INFERENCE-'].ButtonColor[1] != 'green':
            if config['user_status'] == 'Ready' and config['connection_controller_status']=='Connected' and config['connection_muse_status']=='Connected':
                window['notification'].update('')
                window['notification'].update(background_color="lightblue")
                t2 = define_thread(dct=dct_draw,plot=plot)
                window['-INFERENCE-'].update(button_color=("white", "green"))
                app.init_module()
                t2.start()
            else:
                window['notification'].update('NOT ALL REQUIREMENTS READY. MAKE SURE CONTROLLER, MUSE CONNECTED AND USER READY')
                window['notification'].update(background_color="blue")
                window['-INFERENCE-'].update(button_color=("white", "red"))
    if event=='-NOT_INFERENCE-':
        try:
            ##### CHECK ALL STATUS #####
            window['-INFERENCE-'].update(button_color=("white", "blue"))
            dct[config['has_predict']].update(button_color=("black", "lightgrey"))
            t2.join()
            if plot:
                dct_draw['buffer'] = np.zeros((256*2, 13))
        except:
            ##### CHECK ALL STATUS #####
            window['-INFERENCE-'].update(button_color=("white", "blue"))
# Close the window
window.close()
# if __name__ == '__main__':
#     blink = threading.Thread(target=check_muselsl_stream, daemon=True)
#     blink.start()