import argparse
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
from typing import Dict, List, Tuple, Optional
from scipy.signal import savgol_filter

class WristPronationSupinationTracker:
    """
    Трекер для анализа пронации-супинации кистей.
    Отслеживает вращательные движения кисти и сохраняет данные для анализа.
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Инициализация трекера
        
        :param confidence_threshold: порог уверенности для детекции рук
        """
        self.confidence_threshold = confidence_threshold
        
        # Инициализация моделей
        try:
            self.yolo_model = YOLO('yolov11n.pt')
        except FileNotFoundError:
            print("yolov11n.pt не найден, загружаем автоматически...")
            self.yolo_model = YOLO('yolo11n.pt')  # Автоматическая загрузка
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Всегда ищем обе руки
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Параметры повышения точности
        self.CALIBRATION_FRAMES = 25
        self.OUTLIER_JUMP_DEG = 60.0
        self.SAVGOL_WINDOW = 9  # нечетное
        self.SAVGOL_POLY = 2
        self.angle_offset_by_hand: Dict[str, float] = {}
        self._calibration_buffers: Dict[str, List[float]] = {}
        
    def _compute_palm_normal(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Нормаль к плоскости ладони по точкам запястья, оснований указательного и мизинца.
        """
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return normal
        return normal / norm

    def _compute_supination_angle(self, landmarks: np.ndarray) -> float:
        """
        Возвращает угол пронации-супинации как угол подъема нормали ладони относительно оси камеры Z.
        Знак соответствует повороту ладони вверх/вниз.
        """
        normal = self._compute_palm_normal(landmarks)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return 0.0
        nz = float(normal[2])
        nz = np.clip(nz, -1.0, 1.0)
        angle = float(np.arcsin(nz) * 180.0 / np.pi)
        return angle

    def calculate_wrist_angle(self, landmarks: np.ndarray) -> float:
        """
        Расчет угла пронации-супинации кисти
        
        :param landmarks: массив координат ключевых точек кисти
        :return: угол в градусах (-180 до 180)
        """
        # Новая модель: угол определяется нормалью ладони относительно оси камеры Z
        return self._compute_supination_angle(landmarks)
    
    def calculate_rotation_velocity(self, angles: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Расчет скорости вращения
        
        :param angles: массив углов
        :param timestamps: массив временных меток
        :return: массив скоростей вращения
        """
        if len(angles) < 2:
            return np.array([])
            
        velocities = np.zeros_like(angles)
        for i in range(1, len(angles)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                # Учет перехода через 180/-180 градусов
                angle_diff = angles[i] - angles[i-1]
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360
                velocities[i] = angle_diff / dt
                
        return velocities
    
    def detect_rotation_cycles(self, angles: np.ndarray, timestamps: np.ndarray) -> List[Dict]:
        """
        Детекция циклов пронации-супинации
        
        :param angles: массив углов
        :param timestamps: массив временных меток
        :return: список циклов с их характеристиками
        """
        cycles = []
        if len(angles) < 10:  # Минимум данных для анализа
            return cycles
            
        # Сглаживание углов для лучшей детекции (Savitzky–Golay при достаточной длине)
        if len(angles) >= self.SAVGOL_WINDOW:
            smoothed_angles = savgol_filter(angles, self.SAVGOL_WINDOW, self.SAVGOL_POLY, mode='interp')
        else:
            smoothed_angles = self._smooth_signal(angles, window_size=5)
        
        # Поиск экстремумов
        peaks = self._find_peaks(smoothed_angles)
        valleys = self._find_valleys(smoothed_angles)
        
        # Объединение экстремумов в циклы
        extrema = sorted([(i, 'peak', angles[i]) for i in peaks] + 
                        [(i, 'valley', angles[i]) for i in valleys])
        
        for i in range(len(extrema) - 1):
            start_idx, start_type, start_angle = extrema[i]
            end_idx, end_type, end_angle = extrema[i + 1]
            
            if start_type != end_type:  # Полный цикл от пика до впадины или наоборот
                cycle_data = {
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'duration': timestamps[end_idx] - timestamps[start_idx],
                    'amplitude': abs(end_angle - start_angle),
                    'start_angle': start_angle,
                    'end_angle': end_angle,
                    'cycle_type': f"{start_type}_to_{end_type}"
                }
                cycles.append(cycle_data)
                
        return cycles
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Сглаживание сигнала скользящим средним"""
        if len(signal) < window_size:
            return signal
            
        kernel = np.ones(window_size) / window_size
        padded = np.pad(signal, (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(signal)]
    
    def _find_peaks(self, signal: np.ndarray) -> List[int]:
        """Поиск пиков в сигнале"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return peaks
    
    def _find_valleys(self, signal: np.ndarray) -> List[int]:
        """Поиск впадин в сигнале"""
        valleys = []
        for i in range(1, len(signal) - 1):
            if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                valleys.append(i)
        return valleys
    
    def _detect_active_hand(self, wrist_data: List) -> str:
        """
        Определение активной руки на основе амплитуды движений
        
        :param wrist_data: список данных о руках
        :return: 'left' или 'right' - активная рука
        """
        if len(wrist_data) < 10:
            return 'unknown'
        
        # Группируем данные по рукам
        hand_data = {}
        for frame_data in wrist_data:
            hand = frame_data[2]  # hand label
            angle = frame_data[3]  # angle
            if hand not in hand_data:
                hand_data[hand] = []
            hand_data[hand].append(angle)
        
        # Вычисляем амплитуду движений для каждой руки
        hand_amplitudes = {}
        for hand, angles in hand_data.items():
            if len(angles) < 5:
                continue
            angles_array = np.array(angles)
            # Сглаживаем для лучшего анализа
            if len(angles_array) >= 5:
                smoothed = self._smooth_signal(angles_array, window_size=5)
                amplitude = np.max(smoothed) - np.min(smoothed)
                hand_amplitudes[hand] = amplitude
        
        if not hand_amplitudes:
            return 'unknown'
        
        # Выбираем руку с наибольшей амплитудой
        active_hand = max(hand_amplitudes, key=hand_amplitudes.get)
        print(f"Определена активная рука: {active_hand} (амплитуда: {hand_amplitudes[active_hand]:.1f}°)")
        return active_hand

    def analyze_video(self, video_path: str, output_csv: str = "wrist_data.csv", 
                     show_video: bool = True) -> Dict:
        """
        Анализ видео на предмет пронации-супинации кистей
        
        :param video_path: путь к видеофайлу
        :param output_csv: путь для сохранения CSV данных
        :param show_video: показывать ли видео в реальном времени
        :return: словарь с результатами анализа
        """
        # Открытие видеофайла
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Ошибка открытия видеофайла")
        
        # Подготовка для сохранения данных
        wrist_data = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Начинаю анализ видео: {video_path}")
        print(f"FPS: {fps}")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Обнаружение людей с помощью YOLO
            yolo_results = self.yolo_model(frame, classes=[0])
            if yolo_results[0].boxes is not None:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            else:
                boxes = np.array([])
            
            # Обработка области с человеком
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                person_roi = frame[y1:y2, x1:x2]
                results = self.hands.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
            else:
                results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
            
            # Обработка результатов детекции рук
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                handedness_list = results.multi_handedness if hasattr(results, 'multi_handedness') else None
                
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Определение руки
                    hand_label = 'unknown'
                    handed_score = 1.0
                    if handedness_list and len(handedness_list) > idx:
                        try:
                            hand_label = handedness_list[idx].classification[0].label.lower()
                            handed_score = float(handedness_list[idx].classification[0].score)
                        except Exception:
                            hand_label = 'unknown'
                            handed_score = 0.0
                    # Фильтр надежности: игнорируем кадры с низкой уверенностью классификатора руки
                    if handed_score < max(0.3, self.confidence_threshold * 0.8):
                        continue
                    
                    # Получение координат ключевых точек
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x = x1 + landmark.x * (x2 - x1)
                        y = y1 + landmark.y * (y2 - y1)
                        z = landmark.z
                        landmarks.extend([x, y, z])
                    
                    # Расчет угла пронации-супинации
                    landmarks_array = np.array(landmarks).reshape(-1, 3)
                    angle_raw = self.calculate_wrist_angle(landmarks_array)
                    # Калибровка нулевого уровня по первым кадрам
                    buf = self._calibration_buffers.setdefault(hand_label, [])
                    if hand_label in ('left', 'right') and len(buf) < self.CALIBRATION_FRAMES:
                        buf.append(float(angle_raw))
                        if len(buf) == self.CALIBRATION_FRAMES:
                            self.angle_offset_by_hand[hand_label] = float(np.mean(buf))
                    offset = self.angle_offset_by_hand.get(hand_label, 0.0)
                    angle = float(angle_raw - offset)
                    # Отбрасывание одиночных выбросов (слишком резкие скачки)
                    if wrist_data:
                        prev_angle = wrist_data[-1][3] if len(wrist_data[-1]) >= 4 and wrist_data[-1][2] == hand_label else None
                        if prev_angle is not None and abs(angle - prev_angle) > self.OUTLIER_JUMP_DEG:
                            angle = prev_angle
                    
                    # Сохранение данных
                    wrist_data.append([frame_count, timestamp, hand_label, angle] + landmarks)
                    
                    # Визуализация
                    if show_video:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style())
                        
                        # Отображение угла на кадре
                        cv2.putText(frame, f"Angle: {angle:.1f}°", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if show_video:
                cv2.imshow('Wrist Pronation-Supination Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Закрытие ресурсов
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Сохранение данных в CSV
        if wrist_data:
            # Определяем активную руку
            active_hand = self._detect_active_hand(wrist_data)
            
            # Фильтруем данные только для активной руки
            filtered_data = []
            for frame_data in wrist_data:
                if frame_data[2] == active_hand:  # hand label
                    filtered_data.append(frame_data)
            
            if not filtered_data:
                print("Не найдено данных для активной руки")
                return {}
            
            columns = ['frame', 'timestamp', 'hand', 'angle']
            for i in range(21):
                columns.extend([f'hand_{i}_x', f'hand_{i}_y', f'hand_{i}_z'])
            
            df = pd.DataFrame(filtered_data, columns=columns)
            df.to_csv(output_csv, index=False)
            print(f"Данные сохранены в {output_csv} (только активная рука: {active_hand})")
            
            # Анализ циклов только для активной руки
            analysis_results = {}
            hand_data = df[df['hand'] == active_hand]
            angles = hand_data['angle'].values
            timestamps = hand_data['timestamp'].values
            
            # Расчет скорости вращения
            velocities = self.calculate_rotation_velocity(angles, timestamps)
            
            # Детекция циклов
            # Дополнительное сглаживание перед детекцией циклов
            angles_for_cycles = angles
            if len(angles) >= self.SAVGOL_WINDOW:
                angles_for_cycles = savgol_filter(angles, self.SAVGOL_WINDOW, self.SAVGOL_POLY, mode='interp')
            cycles = self.detect_rotation_cycles(np.asarray(angles_for_cycles), timestamps)
            
            analysis_results[active_hand] = {
                'angles': angles.tolist(),
                'timestamps': timestamps.tolist(),
                'velocities': velocities.tolist(),
                'cycles': cycles,
                'num_cycles': len(cycles),
                'mean_amplitude': np.mean([c['amplitude'] for c in cycles]) if cycles else 0,
                'mean_duration': np.mean([c['duration'] for c in cycles]) if cycles else 0
            }
            
            return analysis_results
        else:
            print("Не обнаружено рук на видео")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Анализ пронации-супинации кистей на видео')
    parser.add_argument('--input', required=True, help='Путь к входному видеофайлу')
    parser.add_argument('--output', default="wrist_data.csv", help='Путь для сохранения CSV')
    parser.add_argument('--no-display', action='store_true', help='Не показывать видео')
    parser.add_argument('--confidence', type=float, default=0.3, help='Порог уверенности детекции')
    
    args = parser.parse_args()
    
    tracker = WristPronationSupinationTracker(confidence_threshold=args.confidence)
    
    try:
        results = tracker.analyze_video(
            video_path=args.input,
            output_csv=args.output,
            show_video=not args.no_display
        )
        
        # Сохранение результатов анализа в JSON
        if results:
            with open('wrist_analysis_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print("Результаты анализа сохранены в wrist_analysis_results.json")
            
    except Exception as e:
        print(f"Ошибка при анализе: {str(e)}")

if __name__ == "__main__":
    main()
