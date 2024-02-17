##PROJECT BSS: AUDIO
import pyaudio
import wave
import time
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import json
from decimal import Decimal
import random
import sympy
import math
import soundfile as sf

#======================================================================
##PROBLEM:    1.ACCESSING THE MIKE AND RECORD THE AUDIO AND SAVE IT.
##            2.ALSO GET THE NO OF SECONDS TO BE RECORDED AND SHOW THE TIMER
##STATUS: RUNS SUCCESSFULLY            
def capture_audio():
    # Define the duration of the recording in seconds
    duration =int(input("Enter the duration(seconds):"))
    
    # Set the chunk size and sample rate
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    
    # Create a PyAudio object
    p = pyaudio.PyAudio()
    
    # Open the microphone stream
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    
    print("Recording started...")
    result=0
    time_log=[]
    already_printed=[]
    # Initialize the recording frames and start the timer
    frames = []
    start_time = time.time()
    
    # Loop to capture audio for the specified duration
    while time.time() - start_time < duration:
        data = stream.read(chunk)
        frames.append(data)
        result=(time.time() - start_time)//1
        time_log.append(result)
        unique_list = list(set(time_log))
        if unique_list[-1] in already_printed:
            continue
        else:
            already_printed.append(unique_list[-1])
            print(unique_list[-1])
        
    print("Recording finished...")
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Terminate the PyAudio object
    p.terminate()
    
    # Save the recorded audio to a WAV file
    file_name='Original.wav'
    wf = wave.open(file_name, "wb")
    
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()
    
    print(f"Audio saved to {file_name}.")
#======================================================================

def generate_random_prime():
    while True:
        num = random.randint(500, 1000)
        if sympy.isprime(num):
            return num
#======================================================================
##PROBLEM:      GETTING ALL REQUIREMENT FOR KEY PROCESSING
##STATUS:Runs Successfully

def method_RSA(key):
    # Generate a random prime number
    random_prime_1= generate_random_prime()
    random_prime_2 = generate_random_prime()
    
    p=min(random_prime_1,random_prime_2)
    q=max(random_prime_1,random_prime_2)

    n=p*q
    shy_fn=(p-1)*(q-1)
    possible_nos=[]
    new_possible_nos=[]
    
    for i in range(2,20):
        if ((i%p)!= 0)and((i%q)!= 0) :
            possible_nos.append(i)        
    
    for item in possible_nos:
        gcd = math.gcd(item,shy_fn)
        if gcd == 1:
            new_possible_nos.append(item)        
    return n,key,new_possible_nos,shy_fn
#======================================================================
def encrypt_key(n,key,public_key,shy_fn):
    if key>n:
        print(f"Key size is too large.\n{key} is larger than {n}.")
        return
    encrypt_key=(key**public_key)%n
    return encrypt_key
#======================================================================
def decrypt_key(n,key,public_key,encrypt_key,shy_fn):
    MI_Key,status=RSA_MI(public_key,shy_fn)
    print(f"MI is {MI_Key}.")
    if key>n:
        print(f"Key size is too large.\n{key} is larger than {n}.")
        return
    decrypt_key=(encrypt_key**MI_Key)%n
    return decrypt_key
#======================================================================
##PROBLEM:      TO FIND MI OF THE KEY
##STATUS:Runs Successfully

def RSA_MI(first,second):
    a = first
    m = second    
    if a<-1:
        a=a%m
    inverse = find_inverse(a, m)
    if inverse is None:
        print("MI not exist")
        return 0,-1
    else:
        return inverse,1

#======================================================================

##PROBLEM: ACCESS THE AUDIO FILE AND CONVERT INTO 2D ARRAY
##STATUS:Runs Successfully

def load_audio(input_file):
    sample_rate, audio_data = wavfile.read(input_file)

    # Normalize the audio data to the range [-1, 1]
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))
    
    # Reshape the audio data into a 2D NumPy float array
    num_channels = audio_data.ndim
    num_samples = audio_data.shape[0]
    audio_array = np.reshape(audio_data, (num_samples, num_channels))

    return sample_rate,audio_array
#======================================================================
##PROBLEM: CONVERT THE AUDIO 2D ARRAY TO AUDIO
##STATUS:Runs Successfully

def make_audio(output_file,source_sample_rate,source_audio):
    wavfile.write(output_file,source_sample_rate,source_audio)
    return output_file
#======================================================================
##PROBLEM: GETTING THE INCORRECT ENTRIES
##STATUS:Runs Successfully
def check_accuracy(array_1,array_2):
    entries=[]
    for i in range(outer):
        result = np.not_equal(array_1[i], array_2[i])
        if (result[0]==True)or(result[1]==True):
            entries.append(i)
    return entries
#======================================================================
##PROBLEM: MAKING THE DICTIONARY FOR EACH AND EVERY DECIMAL VALUES WITH INT VALUE
##STATUS:Runs Successfully
def make_dictionary(list_audio):
    outer=len(list_audio)
    inner=len(list_audio[0])
    entries=[]
    for i in range(outer):
        for j in range(inner):
            entries.append(list_audio[i][j])
    unique_list= list(set(entries))
    unique_list.sort()

    ##making dictionary
    keys = unique_list
    values = []
    for i in range(len(keys)):
        values.append(i)
    # Convert list to dictionary
    decimal_int= {k: v for k, v in zip(keys, values)}#{decimal:int}
    int_decimal= {v: k for v, k in zip(values, keys)}#{int:decimal}
    return decimal_int,int_decimal
#======================================================================
##PROBLEM: DOING THE CONVERAION BETWEEN DECIMAL AND INT VALUES
##STATUS:Runs Successfully
def dictionary_exchange(my_list,my_dict):
    for i in range(len(my_list)):
        if my_list[i] in my_dict:
            my_list[i] = my_dict[my_list[i]]
    return my_list
#======================================================================
##PROBLEM: CAONVERTS 2D ARRAY TO 1D ARRAY
##STATUS:Runs Successfully
def array_convertion_1Dfrom2D(array_2D):
    entries_1D=[]
    outer=len(array_2D)
    inner=len(array_2D[0])
    for i in range(outer):
        for j in range(inner):
            entries_1D.append(array_2D[i][j])
    return entries_1D
#======================================================================
##PROBLEM: CAONVERTS 1D ARRAY TO 2D ARRAY
##STATUS:Runs Successfully
def array_convertion_2Dfrom1D(entries_1D):
    entries_2D= np.array(entries_1D).reshape(-1, 2)
    return entries_2D
#======================================================================
##PROBLEM:      KEY MATRIX GENERATION
##STATUS:Runs Successfully
def get_key(mod_value,key):
    key_matrix =[
                    [key+20000,key-2700],
                    [key-21150,key+3066],
                ]
    key_matrix=np.array(key_matrix)
    key_matrix=key_matrix%mod_value
    
    return key_matrix
#======================================================================
##PROBLEM:      TRANSPOSITION MATRIX GENERATION
##STATUS:Runs Successfully
def get_transpose(mod_value,key):
    transpose_matrix=[
                    [key+20000,key-2700],
                    [key-21150,key+3066],
                ]
    
    transpose_matrix=np.array(transpose_matrix)
    transpose_matrix=transpose_matrix%mod_value

    return transpose_matrix
#======================================================================
##PROBLEM:    CHECKING THE PRESENCE OF THE MI AND DETERMINANT
##STATUS:Runs Successfully
def check_matrix(mod_value,test_matrix):
    determinant,i=check_determinant(test_matrix)
    determinant=round(determinant)
    
    determinant,MI,j=check_MI(mod_value,determinant)
    determinant=determinant%mod_value
    if i==1 and j==1:
        print("\nBoth Values are exist.")
        actual_matrix=test_matrix

    else:
        actual_matrix=np.array([
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                ])
        determinant= -1
        MI= -1
    return actual_matrix,MI,determinant

#======================================================================
##PROBLEM:      CHECKING THE PRESENCE OF THE POSITIVE DETERMINANT GREATER THAN ZERO
##STATUS:Runs Successfully
def check_determinant(key_actual):
    arr = np.array(key_actual)
    # calculate the determinant
    det = np.linalg.det(arr)
    
    if det==0:
        return 0,0
    else:
        return det,1
#======================================================================
##PROBLEM:      CHECKING THE PRESENCE OF THE MULTIPLICATIVE INVERSE
##STATUS:Runs Successfully
def check_MI(mod_value,determinant):
    a = int(determinant)
    m = int(mod_value)
    
    if a<-1:
        a=a%mod_value
    inverse = find_inverse(a, m)
    if inverse is None:
        return a,0,0
    else:
        return a,inverse,1

def euclidean_algorithm(a, b):
    if b == 0:
        return a
    else:
        return euclidean_algorithm(b, a % b)

def find_inverse(a, m):
    gcd = euclidean_algorithm(a, m)
    if gcd != 1:
        return None  # The inverse does not exist
    else:
        for i in range(1, m):
            if (a * i) % m == 1:
                return i

#======================================================================
##PROBLEM:    MIXING THE SOURCE WITH GIVEN MATRIX
##STATUS:Runs Successfully
def do_mixing(mod_value,audio_int,matrix):
    
    print(f"\nInput is {audio_int[0]} and ")
    print(f"The mod value is {mod_value}.")
    print(matrix)
    outer=len(audio_int)
    inner=len(audio_int[0])
    
    for i in range(outer):
        A=np.array(audio_int[i])
        B=np.array(matrix)

        A=A.tolist()
        B=B.tolist()
        array1=A
        array2=B
        result = [0, 0]  # Initialize the result as a 1x2 array of zeros

        result[0] = array1[0] * array2[0][0] + array1[1] * array2[1][0]
        result[1] = array1[0] * array2[0][1] + array1[1] * array2[1][1]
    
        result[0]=result[0]%mod_value
        result[1]=result[1]%mod_value
        result=np.array(result)
        audio_int[i]=result
    print(f"\nOutput is {audio_int[0]}.")
    return audio_int

#======================================================================

##PROBLEM:    FIND THE INVERSE OF THE GIVEN MATRIX
##STATUS:Runs Successfully
def find_inverse_matrix(mod_value,multi_no,matrix):

    adjoint=find_adjoint(matrix)
    matrix=multi_no*adjoint
    matrix=matrix%mod_value
    
    return matrix
#======================================================================
##PROBLEM:    FIND THE ADJOINT
##STATUS:Runs Successfully
def find_adjoint(matrix):
    A=matrix
    # Calculate the matrix of minors
    minors = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            sub_matrix = A[np.array(list(range(i))+list(range(i+1,2)))[:,np.newaxis],
                           np.array(list(range(j))+list(range(j+1,2)))]
            minors[i, j] = (-1)**(i+j) * np.linalg.det(sub_matrix)
    
    # Transpose the matrix of minors to get the matrix of cofactors
    cofactors = minors.T
        
    # Calculate the adjoint of the matrix
    adj_A = cofactors
    return adj_A
#======================================================================
##PROBLEM:  DOING THE ENCRYPTION PROCESS AND RETURNING AUDIO ARRAY WITH INT VALUE
##STATUS:Runs Successfully
def do_encryption(mod_value,source_audio_int_2D):
    print("\t\t\t!!!!     ENCRYPTION PROCESS     !!!!")
    print(f"The mod value is {mod_value}.")

    #Getting the key matrix
    key=int(input("Please enter the key:"))
    key_matrix=get_key(mod_value,key)
    
    temp=key
    print(f"The initial key value is {key}.")

    print("\nGetting the key matrix......")
    key_final,MI_key,determinant_key=check_matrix(mod_value,key_matrix)
    check=0
    while(MI_key == -1):
        start_time = time.time()
        result=time.time() - start_time
        while (result < 2)and(MI_key == -1):
            result=time.time() - start_time
            temp=temp-1
            key_matrix=get_key(mod_value,temp)
            key_final,MI_key,determinant_key=check_matrix(mod_value,key_matrix)
        temp=key
        check=1
        mod_value=mod_value-1
    if check==1:
        print("!!!!!  ALERT  !!!!!")
        mod_value=mod_value+1
        print(f"Mod Value is reduced from {mod_value+1} to {mod_value}.")
    print(f"The final key value is {temp}.")
    print("\nFINAL KEY IS")        
    print(key_final)

    print(f"The Determinant of {key_final} is {determinant_key}.")
    print(f"The MI_key of {determinant_key} is {MI_key}.")
    
    print("\n\nGetting the transpose matrix......")
    print(f"Mod value for transpose is {mod_value}.")
    MI_transpose= -1
    while MI_transpose== -1:
        num = random.randint(500, 1000)
        transpose_matrix=get_transpose(mod_value,num)
        transpose_final,MI_transpose,determinant_transpose=check_matrix(mod_value,transpose_matrix)
    print(f"Num Value is {num}.")
    print("\n\nFINAL TRANSPOSE  IS")                
    print(transpose_final)

    print(f"The Determinant of {transpose_final} is {determinant_transpose}.")
    print(f"The MI_key of {determinant_transpose} is {MI_transpose}.")

    print("\n\nMixing the Source and Key......")
    source_audio_int_2D=do_mixing(mod_value,source_audio_int_2D,key_final)
    print("Finished")

    print("\n\nMixing the Source Key and Transpose Matrix......")
    source_audio_int_2D=do_mixing(mod_value,source_audio_int_2D,transpose_final)
    print("Finished")

    #Getting the Public Key
    n,key,new_possible_nos,shy_fn=method_RSA(key)    
    
    status= -1
    while status== -1:
        public_key= random.choice(new_possible_nos)
        MI_Key,status=RSA_MI(public_key,shy_fn)
        new_possible_nos.remove(public_key)
        
    print(f"\nPublic Key is {public_key}.")
    
    Encrypt_key=encrypt_key(n,key,public_key,shy_fn)
    print(f"Encrypted key is {Encrypt_key}.")
    
    return n,key,public_key,shy_fn,Encrypt_key,source_audio_int_2D,num,transpose_final
#======================================================================
##PROBLEM:  DOING THE DECRYPTION PROCESS AND RETURNING AUDIO ARRAY WITH INT VALUE
##STATUS:Runs Successfully
def do_decryption(n,public_key,encrypt_key,shy_fn,mod_value,source_audio_int_2D,num,transpose_final):
    print("\t\t\t!!!!     DECRYPTION PROCESS     !!!!")
    print(f"The mod value is {mod_value}.")
    
    #Getting the key matrix
    encrypt_key=int(input("Please enter the key:"))
    print(f"\nEncrypted Key is {encrypt_key}.")
    Decrypt_key=decrypt_key(n,key,public_key,encrypt_key,shy_fn)
    print(f"Decrypted key is {Decrypt_key}.")

    key_matrix=get_key(mod_value,Decrypt_key)
    
    temp=Decrypt_key

    print("\nGetting the key matrix......")
    key_final,MI_key,determinant_key=check_matrix(mod_value,key_matrix)
    check=0
    while(MI_key == -1):
        start_time = time.time()
        result=time.time() - start_time
        while (result < 2)and(MI_key == -1):
            result=time.time() - start_time
            temp=temp-1
            key_matrix=get_key(mod_value,temp)
            key_final,MI_key,determinant_key=check_matrix(mod_value,key_matrix)
        temp=Decrypt_key
        check=1
        mod_value=mod_value-1
    if check==1:
        print("!!!!!  ALERT  !!!!!")
        mod_value=mod_value+1
        print(f"Mod Value is reduced from {mod_value+1} to {mod_value}.")
    print(f"The final key value is {temp}.")
    print("\nFINAL KEY IS")        
    print(key_final)

    print(f"The Determinant of {key_final} is {determinant_key}.")
    print(f"The MI_key of {determinant_key} is {MI_key}.")
    
    print("\n\nGetting the transpose matrix......")
    transpose_final,MI_transpose,determinant_transpose=check_matrix(mod_value,transpose_final)

    print(f"Num Value is {num}.")
    print("FINAL TRANSPOSE  IS")                
    print(transpose_final)

    print(f"The Determinant of {transpose_final} is {determinant_transpose}.")
    print(f"The MI_key of {determinant_transpose} is {MI_transpose}.")


    print("\nCalculating the Inverse..........")

    print("\nKEY INVERSE")
    key_inverse=find_inverse_matrix(mod_value,MI_key,key_final)

    for i in range(2):
        for j in range(2):
            key_inverse[i][j]=round(key_inverse[i][j])
    print(key_inverse)

    print("\nTRANSPOSE INVERSE")
    transpose_inverse=find_inverse_matrix(mod_value,MI_transpose,transpose_final)
    
    for i in range(2):
        for j in range(2):
            transpose_inverse[i][j]=round(transpose_inverse[i][j])
    print(transpose_inverse)

    print("\nDemixing the Source Key and Transpose Inverse......")
    source_audio_int_2D=do_mixing(mod_value,source_audio_int_2D,transpose_inverse)
    print("Finished")

    print("Demixing the Source and Key Inverse......")
    source_audio_int_2D=do_mixing(mod_value,source_audio_int_2D,key_inverse)
    print("Finished")

    return source_audio_int_2D
#======================================================================
##PROBLEM: TO DISPLAY THE ERROR ENTRIES ALONG WITH IT'S ENTRIES
##STATUS:Runs Successfully
def display_error_list(incorrect_entries,length,source_audio_int_2D,source_audio_int_2D_back_up):
    for i in range(length):
        x=incorrect_entries[i]
        print(f"When 'X' value is {x},")
        print(f"{source_audio_int_2D[x]} should be {source_audio_int_2D_back_up[x]}.\n")
#======================================================================
print("\t\t\t!!!!!!     BLIND SOURCE SEPARATION(AUDIO)     !!!!!! ")
print("Press Enter to Capture Audio.")
input()
capture_audio()

# Read the audio file
input_file='Original.wav'
 
source_sample_rate,source_audio_float_2D=load_audio(input_file)
source_sample_rate,source_audio_float_2D_backup=load_audio(input_file)

outer=len(source_audio_float_2D)
inner=len(source_audio_float_2D[0])
print(f"The length of the source audio is {outer}x{inner}.")

#making dectionary
decimal_int,int_decimal=make_dictionary(source_audio_float_2D)

print("\nDICTIONARY (decimal:int)")
print(f"The length of the decimal_int is {len(decimal_int)}.")

print("\nDICTIONARY (int:decimal)")
print(f"The length of the int_decimal is {len(int_decimal)}.")


#Dictionary exchange(decimal to int)
source_audio_float_1D=array_convertion_1Dfrom2D(source_audio_float_2D)
source_audio_int_1D=dictionary_exchange(source_audio_float_1D,decimal_int)
source_audio_int_2D =array_convertion_2Dfrom1D(source_audio_int_1D)
source_audio_int_2D_back_up=array_convertion_2Dfrom1D(source_audio_int_1D)
mod_value=max(source_audio_int_1D)
mod_value=mod_value+1
    
##Encryption should be carried out
n,key,public_key,shy_fn,Encrypt_key,source_audio_int_2D,num,transpose_final=do_encryption(mod_value,source_audio_int_2D)
        
#Dictionary exchange(int to decimal)
source_audio_int_1D=array_convertion_1Dfrom2D(source_audio_int_2D)
source_audio_float_1D=dictionary_exchange(source_audio_int_1D,int_decimal)
source_audio_float_2D =array_convertion_2Dfrom1D(source_audio_float_1D)

#Saving back to audio
output_file=make_audio("Encrypted.wav",source_sample_rate,source_audio_float_2D)
print(f"\nAudio saved to File name '{output_file}' successfully.")

#Dictionary exchange(decimal to int)
source_audio_float_1D=array_convertion_1Dfrom2D(source_audio_float_2D)
source_audio_int_1D=dictionary_exchange(source_audio_float_1D,decimal_int)
source_audio_int_2D =array_convertion_2Dfrom1D(source_audio_int_1D)

##Decryption should be carried out
source_audio_int_2D=do_decryption(n,public_key,encrypt_key,shy_fn,mod_value,source_audio_int_2D,num,transpose_final)

#Dictionary exchange(int to decimal)
source_audio_int_1D=array_convertion_1Dfrom2D(source_audio_int_2D)
source_audio_float_1D=dictionary_exchange(source_audio_int_1D,int_decimal)
source_audio_float_2D =array_convertion_2Dfrom1D(source_audio_float_1D)

#Saving back to audio
output_file=make_audio("Decrypted.wav",source_sample_rate,source_audio_float_2D)
print(f"\nAudio saved to File name '{output_file}' successfully.")

#Checking the accuracy
print("\t\t\t!!!!    ACCURACY CHECKING     !!!!")
print("After Encryption Accuracy:")
incorrect_entries=check_accuracy(source_audio_float_2D,source_audio_float_2D_backup)

numerator=len(source_audio_float_2D)-len(incorrect_entries)
denominator=len(source_audio_int_2D)
percentage=(numerator/denominator)*100
print(f"Accuracy percentage is {percentage}.")

incorrect_entries=check_accuracy(source_audio_int_2D,source_audio_int_2D_back_up)

length=len(incorrect_entries)
if length==0:
    print(f"Total no of incorrect entries is {length}.")
    print("\n\t\t!!!!Hurrah!!!!\nThere is no incorrect entries!!!!!!")
else:
    print(f"\nOops!!!! There are incorrect entries.\nTotal no of incorrect entries is {length}.")
    choice=input("\nDo you want to see the incorrect entries?\nType 'yes' or 'no':")
    if choice =='yes':
        display_error_list(incorrect_entries,length,source_audio_int_2D,source_audio_int_2D_back_up)

print("\nDoing the cryptanalysis......")
# Load the original and reconstructed audio signals
orig_signal, orig_fs = sf.read('Original.wav')
recon_signal, recon_fs = sf.read('Decrypted.wav')

# Make sure both signals have the same length
min_len = min(len(orig_signal), len(recon_signal))
orig_signal = orig_signal[:min_len]
recon_signal = recon_signal[:min_len]

# Calculate the signal power and noise power
signal_power = np.mean(orig_signal ** 2)
noise_power = np.mean((orig_signal - recon_signal) ** 2)

print(f"signal_power is {signal_power}.")
print(f"noise_power is {noise_power}.")
# Calculate the SNR in decibels
snr = 10 * np.log10(signal_power / noise_power)
print('SNR:', snr, 'dB')

print("\t\t\t!!!!!!     END OF THE PROGRAM     !!!!!!")
#======================================================================
