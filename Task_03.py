import numpy 
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

class Harmonic:
    '''
    Источник, создающий гармонический сигнал
    '''
    def __init__(self, Nl, eps=1.0, mu=1.0, phi_0=None, Sc=1.0, magnitude=1.0):
        '''
        magnitude - максимальное значение в источнике;
        Nl - количество отсчетов на длину волны;
        Sc - число Куранта.
        '''
        
        self.Nl = Nl
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

        if phi_0 is None:
            self.phi_0 = -2 * numpy.pi / Nl
        else:
            self.phi_0 = phi_0

    def getField(self, m, q):
        return self.magnitude * numpy.sin(2 * numpy.pi / self.Nl *
                (self.Sc * q - m * numpy.sqrt(self.mu * self.eps))+ self.phi_0)

def Spectr(dt: float, probe: float):
    '''
    Функция для расчета спектра
    '''
    size = 2 ** 15

    # Шаг дискретизации по частоте
    df = 1 / (size * dt)

    # Нахождение спектра
    f = numpy.arange(-size / 2 * df, size / 2 * df, df)
    spectr_harmonic = fftshift(numpy.abs(fft(probe, size)))

    # Построение графика
    plt.plot(f, spectr_harmonic / numpy.max(spectr_harmonic))
    plt.xlim(0, 1e9)
    plt.ylim(0, 1)
    plt.xlabel('f, [Гц]')
    plt.ylabel('|P / Pmax|')
    plt.grid()
    plt.show()  
    

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 1300

    # Размер области моделирования вдоль оси X в метрах
    X = 5.0

    #Размер ячейки разбиения
    dx = 1e-2

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)

    # Датчики для регистрации поля
    probesPos = [sourcePos - 100]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[:] = 2.5

    # Магнитная проницаемость
    mu = numpy.ones(maxSize)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize)

    source = Harmonic(100.0, eps[sourcePos], mu[sourcePos])

    # Sc' для правой границы
    Sc1Right = Sc / numpy.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = numpy.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = numpy.zeros(3)
    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.0
    display_ymax = 2.0

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Граничные условия для поля Н (слева)
        Hy[0] = 0
        
        # Расчет компоненты поля H
        Ez_shift = Ez[:-1]
        Hy[1:] = Hy[1:] + (Ez_shift - Ez[1:]) * Sc / (W0 * mu[1:])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        # Создается источник, бегущий влево, для отражения от левой стенки
        Hy[sourcePos] -= Sc / (W0 * mu[sourcePos]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[1:]
        Ez[:-1] = Ez[:-1] + (Hy[:-1] - Hy_shift) * Sc * W0 / eps[:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos - 1] += (Sc / (numpy.sqrt(eps[sourcePos - 1] * mu[sourcePos - 1])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]
        
        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 5 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -2.0, 2.0, dt)

    # Отображение спектра сигнала
    Spectr(dt, probe.E)

