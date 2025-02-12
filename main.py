
import time
import cv2
import numpy as np
import smbus2 as smbus

# ============================
# Класс для работы с PCA9685
# ============================
class PCA9685:
    def __init__(self, bus_num=1, address=0x60):
        self.bus = smbus.SMBus(bus_num)
        self.address = address
        self.reset()
        self.set_pwm_freq(1000)  # Установка частоты ШИМ 1 кГц

    def reset(self):
        self.bus.write_byte_data(self.address, 0x00, 0x00)

    def set_pwm_freq(self, freq):
        prescaleval = 25000000.0  # 25MHz
        prescaleval /= 4096.0     # 12-bit разрешение
        prescaleval /= float(freq)
        prescaleval -= 1.0
        prescale = int(prescaleval + 0.5)

        old_mode = self.bus.read_byte_data(self.address, 0x00)
        new_mode = (old_mode & 0x7F) | 0x10  # переход в режим сна
        self.bus.write_byte_data(self.address, 0x00, new_mode)
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        self.bus.write_byte_data(self.address, 0x00, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, 0x00, old_mode | 0x80)

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4 * channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4 * channel, on >> 8)
        self.bus.write_byte_data(self.address, 0x08 + 4 * channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4 * channel, off >> 8)

# ====================================
# Класс управления моторами (движение)
# ====================================
class MotorDriver:
    def __init__(self):
        self.pca = PCA9685(address=0x60)  # Убедитесь, что адрес соответствует вашей плате

        # Конфигурация каналов (адаптируйте под вашу схему подключения)
        # Предполагается, что мотор A – левый, мотор B – правый
        self.AIN1 = 10  # Направление 1 левого мотора
        self.AIN2 = 9   # Направление 2 левого мотора
        self.PWMA = 8   # ШИМ канал левого мотора
        self.BIN1 = 11  # Направление 1 правого мотора
        self.BIN2 = 12  # Направление 2 правого мотора
        self.PWMB = 13  # ШИМ канал правого мотора

    def motor_control(self, motor, in1, in2, speed):
        """Управление мотором
           motor: 'A' или 'B'
           in1, in2: 0 или 1 — логические уровни для направления
           speed: значение от 0.0 до 1.0 для ШИМ
        """
        if motor not in ['A', 'B']:
            raise ValueError("Motor must be 'A' or 'B'")
        if not (0.0 <= speed <= 1.0):
            raise ValueError("Speed must be between 0.0 and 1.0")

        if motor == 'A':
            self.pca.set_pwm(self.AIN1, 0, 4095 if in1 else 0)
            self.pca.set_pwm(self.AIN2, 0, 4095 if in2 else 0)
            self.pca.set_pwm(self.PWMA, 0, int(4095 * speed))
        elif motor == 'B':
            self.pca.set_pwm(self.BIN1, 0, 4095 if in1 else 0)
            self.pca.set_pwm(self.BIN2, 0, 4095 if in2 else 0)
            self.pca.set_pwm(self.PWMB, 0, int(4095 * speed))

    def forward(self, motor, speed=0.0):
        self.motor_control(motor, 0, 1, speed)

    def backward(self, motor, speed=0.0):
        self.motor_control(motor, 1, 0, speed)

    def stop(self, motor):
        self.motor_control(motor, 0, 0, 0)

    def set_speeds(self, left_speed, right_speed):
        """
        Устанавливает скорости обоих моторов для движения вперед.
        left_speed и right_speed — значения от 0.0 до 1.0.
        """
        self.forward('A', left_speed)
        self.forward('B', right_speed)

    def stop_all(self):
        self.stop('A')
        self.stop('B')

# =======================
# Класс PID-регулятора
# =======================
class PID:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint  # Желаемое значение (центр изображения => 0)
        self.last_error = 0
        self.integral = 0
        self.last_time = time.monotonic()

    def update(self, measurement):
        current_time = time.monotonic()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 1e-3
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        self.last_time = current_time
        return output

# ===================================================
# Функция для обработки изображения и вычисления ошибки
# ===================================================
def get_line_error(frame):
    """
    Обрабатывает кадр для поиска линии.
    Возвращает:
      - error: смещение линии относительно центра ROI (положительное, если линия справа)
      - thresh: бинаризованное изображение области интереса (для отладки)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Берём нижнюю часть кадра, где ожидаем увидеть линию
    roi = gray[int(height*0.7):height, :]

    # Применяем пороговую фильтрацию (предполагается, что линия темная на светлом фоне)
    _, thresh = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)

    # Вычисляем моменты для определения центра масс линии
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
    else:
        cx = width // 2  # Если линия не обнаружена, считаем её центр

    error = cx - (width // 2)
    return error, thresh

# ===========================================
# Основной цикл для слежения робота за линией
# ===========================================
def line_following():
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    driver = MotorDriver()
    pid = PID(kp=0.001, ki=0.00, kd=0.00, setpoint=0)

    base_speed = 0.1  # Базовая скорость движения вперед
    max_correction = 0.2  # Ограничение максимального отклонения скорости
    min_speed = 0.1       # Минимальная скорость мотора, чтобы всегда двигаться вперед
    max_speed = 0.3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            error, thresh = get_line_error(frame)
            correction = pid.update(error)

            # Ограничиваем коррекцию, чтобы не снижать скорость моторов ниже минимума
            correction = max(-max_correction, min(max_correction, correction))

            # Корректировка скорости: если линия смещена, один мотор работает быстрее, другой медленнее,
            # но оба мотора всегда движутся вперед
            left_speed = base_speed - correction
            right_speed = base_speed + correction

            # Ограничение скоростей в диапазоне [min_speed, max_speed]
            left_speed = max(min_speed, min(max_speed, left_speed))
            right_speed = max(min_speed, min(max_speed, right_speed))

            driver.set_speeds(left_speed, right_speed)

            # Для отладки показываем изображения
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Прерывание пользователем")

    finally:
        driver.stop_all()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    line_following()