import pygame
import sounddevice as sd
import numpy as np
import pyaudio
import matplotlib.pyplot as plt

"""
Notas musicales
A La
B Si
C Do
D Re
E Mi
F Fa
G Sol
# Sostenido (medio tono adelante)
"""

pygame.init()

import os
# Posicion ventana inicial pygame
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (593,50) #573

win = pygame.display.set_mode((620, 590)) #1100
image = pygame.image.load('guitar3.jpg')
win.fill((240,240,100))

class button(): 
	def __init__(self, color, x, y, width, height, text=''): 
		self.color = color 
		self.x = x 
		self.y = y 
		self.width = width 
		self.height = height 
		self.text = text

	def draw(self,win,outline=None):
		if outline:
			pygame.draw.ellipse(win, outline, (self.x - 2, self.y - 2, self.width + 4, self.height + 4), 0)

		pygame.draw.ellipse(win, self.color, (self.x, self.y, self.width, self.height), 0) 
		if self.text != '': 
			font = pygame.font.SysFont('comicsans', 60) 
			text = font.render(self.text, 1, (0,0,0)) 
			win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

	def isOver(self, pos):
		if pos[0] > self.x and pos[0] < self.x + self.width: 
			if pos[1] > self.y and pos[1] < self.y + self.height: 
				return True
		return False


class note_window():
	def __init__(self, font_size, color, x, y, width, height, text=''):
		self.font_size = font_size	
		self.color = color
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.text = text
		
	def draw(self, win):
		if self.text != '':
			pygame.draw.rect(win, self.color, (self.x - 2, self.y - 2, self.width + 4, self.height + 4), 0)
			font = pygame.font.SysFont('Comic Sans MS', self.font_size)
			text = font.render(self.text, 1, (200,200,200))
			win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

def readWindow():
	win.fill((50,50,50))
	win.blit(image, [-50,0])
	E2_Button.draw(win, (250,250,250))
	A2_Button.draw(win, (250,250,250))
	D3_Button.draw(win, (250,250,250))
	G3_Button.draw(win, (250,250,250))
	B3_Button.draw(win, (250,250,250))
	E4_Button.draw(win, (250,250,250))
	
E2_Button = button((237,167,96), 2, 295, 100, 100, 'E2')
A2_Button = button((237,167,96), 2, 175, 100, 100, 'A2')
D3_Button = button((237,167,96), 2, 55, 100, 100, 'D3')
G3_Button = button((237,167,96), 518, 55, 100, 100, 'G3')
B3_Button = button((237,167,96), 518, 175, 100, 100, 'B3')
E4_Button = button((237,167,96), 518, 295, 100, 100, 'E4')


nota_baja = 40       # E2 Nota mas baja (sexta cuerda)
nota_alta = 64       # E4 Nota mas alta
fs = 44100           # 44.1kHz Frecuencia de muestreo
tam_frame = 2048     # Muestras por frame
frames_por_fft = 16 
frec_ref = 329.63	 # Frecuencia de referencia E4 (n=64)

muestras_FFT = tam_frame * frames_por_fft #32768 Muestras por cada transformada
freq_step = float(fs) / muestras_FFT #1.34

note_names = 'E F F# G G# A A# B C C# D D#'.split()

def freq_to_number(f): 
    return 64 + 12 * np.log2(f / frec_ref)

def number_to_freq(n): 
    return frec_ref * 2.0**((n - 64) / 12.0)

def note_name(n):
    return note_names[n % nota_baja % len(note_names)] + str(int(n / 12 - 1))

def note_to_fftbin(n): 
    return number_to_freq(n) / freq_step

imin = int(np.floor(note_to_fftbin(nota_baja - 1))) #57
imax = int(np.ceil(note_to_fftbin(nota_alta + 1)))  #367

buf = np.zeros(muestras_FFT, dtype=np.float32)
num_frames = 0


audio = pyaudio.PyAudio() #Crear instancia de pyaudio
#Se crea el stream de pyaudio
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=tam_frame)
stream.start_stream()

#Enventanado BLackman
window = 0.5*np.blackman(muestras_FFT)

#plt.style.use('ggplot')
plt.style.use('dark_background')
fig = plt.figure(figsize=(5, 4))

fig.show()
plt.title('Espectro de magnitud')
plt.ylabel('|X(n)|')
plt.xlabel('frequency[Hz]')

note2 = ''
freq2 = 0
while stream.is_active():
	buf[:-tam_frame] = buf[tam_frame:]
	buf[-tam_frame:] = np.fromstring(stream.read(tam_frame), np.int16)

	#Se calcula la transformada (rfft) para solo obtener los valores no negativos de frecuencia
	transform = np.fft.rfft(buf * window)
	#Frecuencia detectada (punto máximo de la transformada)
	freq = (np.abs(transform[imin:imax]).argmax() + imin) * freq_step
	
	n = freq_to_number(freq)
	n0 = int(round(n))
	num_frames += 1
	
	
	if num_frames>=frames_por_fft and max(transform)>4000000:

		print('Frecuencia: ', round(freq,2), "	Nota: ",note_name(n0), round(n-n0,2) )
		
		timeX = np.arange(0, (fs/2)+1, fs/32768)
		timeX = timeX[:400]

        #Si la frecuencia detectada es la misma que la anterior, entonces no refrescar grafica
		if max(transform) > 1000000 and note_name(n0) != note2:
			note2 = note_name(n0)
			freq2 = freq
			Spectrum = abs(transform)
			plt.plot(timeX, Spectrum[:400],'cyan')
			plt.title('Spectrum')
			plt.ylabel('|X(n)|')
			plt.xlabel('frequency[Hz]')
			plt.draw()
			plt.pause(0.2)
			fig.clear()
	else:
		pass
	
	
	
	readWindow()
	pygame.display.update()
	note = note_window(100, (117,56,26), 265, 160, 100, 50, note2)
	note.draw(win)
	f = note_window(30, (117,56,26), 265, 80, 100, 50, str(round(freq2,2))+"Hz")
	f.draw(win)

	pygame.display.update()
	for event in pygame.event.get():
		pos = pygame.mouse.get_pos()
		if event.type == pygame.QUIT:
			run = False
			pygame.quit()
			quit()
		if event.type == pygame.MOUSEBUTTONDOWN:
			if E2_Button.isOver(pos):
				sd.play(2.0*np.sin(2*np.pi*82.41*np.arange(44100)/44100), samplerate=44100, blocking=True)
				print("E2 Sexta cuerda")
			elif A2_Button.isOver(pos):
				sd.play(2.0*np.sin(2*np.pi*110.00*np.arange(44100)/44100), samplerate=44100, blocking=True)
				print("A2 Quinta cuerda")
			elif D3_Button.isOver(pos):
				sd.play(2.0*np.sin(2*np.pi*146.83*np.arange(44100)/44100), samplerate=44100, blocking=True)
				print("D3 Cuarta cuerda")
			elif G3_Button.isOver(pos):
				sd.play(2.0*np.sin(2*np.pi*196.00*np.arange(44100)/44100), samplerate=44100, blocking=True)
				print("G3 Tercera cuerda")
			elif B3_Button.isOver(pos):
				sd.play(2.0*np.sin(2*np.pi*246.94*np.arange(44100)/44100), samplerate=44100, blocking=True)
				print("B3 Segunda cuerda")
			elif E4_Button.isOver(pos):
				sd.play(2.0*np.sin(2*np.pi*329.63*np.arange(44100)/44100), samplerate=44100, blocking=True)
				print("E4 Primera cuerda")
		if event.type == pygame.MOUSEMOTION:
			if E2_Button.isOver(pos):
				E2_Button.color = (166,221,234)
			elif A2_Button.isOver(pos):
				A2_Button.color = (166,221,234)
			elif D3_Button.isOver(pos):
				D3_Button.color = (166,221,234)
			elif G3_Button.isOver(pos):
				G3_Button.color = (166,221,234)
			elif B3_Button.isOver(pos):
				B3_Button.color = (166,221,234)
			elif E4_Button.isOver(pos):
				E4_Button.color = (166,221,234)
			else:
				E2_Button.color = (1,169,204)
				A2_Button.color = (1,169,204)
				D3_Button.color = (1,169,204)
				G3_Button.color = (1,169,204)
				B3_Button.color = (1,169,204)
				E4_Button.color = (1,169,204)