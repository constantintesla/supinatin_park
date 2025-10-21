import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"

class PronationSupinationAnalyzer:
    """
    Анализатор пронации-супинации кистей.
    Оценивает качество выполнения вращательных движений по шкале 0-4.
    """
    
    def __init__(self, csv_path: str, fps: float = 30):
        """
        Инициализация анализатора
        
        :param csv_path: путь к CSV с данными трекинга
        :param fps: кадровая частота видео
        """
        self.data = pd.read_csv(csv_path)
        self.fps = fps
        self.hand_results = {Hand.LEFT: None, Hand.RIGHT: None}
        
        # Пороговые значения для классификации
        self.MIN_AMPLITUDE_THRESHOLD = 30.0  # Минимальная амплитуда в градусах
        self.MIN_CYCLES_THRESHOLD = 3  # Минимальное количество циклов
        self.RHYTHM_CONSISTENCY_THRESHOLD = 0.7  # Порог ритмичности
        self.SPEED_CONSISTENCY_THRESHOLD = 0.6  # Порог стабильности скорости
        self.AMPLITUDE_CONSISTENCY_THRESHOLD = 0.8  # Порог стабильности амплитуды
        
    def _hand_from_label(self, label: str) -> Hand:
        """Определение руки по метке"""
        label = (label or '').strip().lower()
        return Hand.LEFT if label == 'left' else Hand.RIGHT
    
    def _smooth_signal(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        """Сглаживание сигнала скользящим средним"""
        if window <= 1 or len(values) == 0:
            return values
        window = min(window, max(1, len(values)))
        kernel = np.ones(window, dtype=float) / window
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(values)]
    
    def _detect_rotation_cycles(self, angles: np.ndarray, timestamps: np.ndarray) -> List[Dict]:
        """
        Детекция циклов пронации-супинации
        
        :param angles: массив углов
        :param timestamps: массив временных меток
        :return: список циклов с характеристиками
        """
        cycles = []
        if len(angles) < 10:
            return cycles
        
        # Сглаживание углов
        smoothed_angles = self._smooth_signal(angles, window=5)
        
        # Поиск экстремумов
        peaks, _ = signal.find_peaks(smoothed_angles, distance=int(self.fps * 0.5))
        valleys, _ = signal.find_peaks(-smoothed_angles, distance=int(self.fps * 0.5))
        
        # Объединение экстремумов
        extrema = []
        for peak in peaks:
            extrema.append((peak, 'peak', angles[peak]))
        for valley in valleys:
            extrema.append((valley, 'valley', angles[valley]))
        
        extrema.sort(key=lambda x: x[0])
        
        # Построение циклов
        for i in range(len(extrema) - 1):
            start_idx, start_type, start_angle = extrema[i]
            end_idx, end_type, end_angle = extrema[i + 1]
            
            if start_type != end_type:  # Полный цикл
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
    
    def _calculate_rhythm_consistency(self, cycles: List[Dict]) -> float:
        """Расчет ритмичности движений"""
        if len(cycles) < 2:
            return 0.0
        
        durations = [cycle['duration'] for cycle in cycles]
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        if mean_duration == 0:
            return 0.0
        
        consistency = 1.0 - (std_duration / mean_duration)
        return max(0.0, min(1.0, consistency))
    
    def _calculate_speed_consistency(self, angles: np.ndarray, timestamps: np.ndarray) -> float:
        """Расчет стабильности скорости движений"""
        if len(angles) < 2:
            return 0.0
        
        # Расчет скорости вращения
        velocities = []
        for i in range(1, len(angles)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                angle_diff = angles[i] - angles[i-1]
                # Учет перехода через 180/-180 градусов
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360
                velocities.append(abs(angle_diff) / dt)
        
        if len(velocities) < 2:
            return 0.0
        
        velocities = np.array(velocities)
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        
        if mean_velocity == 0:
            return 0.0
        
        consistency = 1.0 - (std_velocity / mean_velocity)
        return max(0.0, min(1.0, consistency))
    
    def _calculate_amplitude_consistency(self, cycles: List[Dict]) -> float:
        """Расчет стабильности амплитуды движений"""
        if len(cycles) < 2:
            return 0.0
        
        amplitudes = [cycle['amplitude'] for cycle in cycles]
        mean_amplitude = np.mean(amplitudes)
        std_amplitude = np.std(amplitudes)
        
        if mean_amplitude == 0:
            return 0.0
        
        consistency = 1.0 - (std_amplitude / mean_amplitude)
        return max(0.0, min(1.0, consistency))
    
    def _detect_movement_quality(self, angles: np.ndarray, timestamps: np.ndarray) -> Dict:
        """
        Детекция качества движений:
        - плавность
        - ритмичность
        - остановки и задержки
        - уменьшение амплитуды
        """
        if len(angles) < 10:
            return {
                'smoothness': 0.0,
                'rhythm_consistency': 0.0,
                'stops_detected': 0,
                'amplitude_decrease': 0.0
            }
        
        # Сглаживание для анализа плавности
        smoothed_angles = self._smooth_signal(angles, window=3)
        
        # Расчет производной для анализа плавности
        derivatives = np.abs(np.diff(smoothed_angles))
        smoothness = 1.0 - np.mean(derivatives) / 180.0  # Нормализация
        smoothness = max(0.0, min(1.0, smoothness))
        
        # Детекция остановок (низкая скорость движения)
        velocities = []
        for i in range(1, len(angles)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                angle_diff = abs(angles[i] - angles[i-1])
                velocities.append(angle_diff / dt)
        
        if velocities:
            velocity_threshold = np.percentile(velocities, 20)  # Нижний квартиль
            stops = sum(1 for v in velocities if v < velocity_threshold)
            stops_ratio = stops / len(velocities)
        else:
            stops_ratio = 0.0
        
        # Анализ уменьшения амплитуды во времени
        cycles = self._detect_rotation_cycles(angles, timestamps)
        amplitude_decrease = 0.0
        if len(cycles) >= 3:
            early_cycles = cycles[:len(cycles)//3]
            late_cycles = cycles[-len(cycles)//3:]
            
            early_amplitude = np.mean([c['amplitude'] for c in early_cycles])
            late_amplitude = np.mean([c['amplitude'] for c in late_cycles])
            
            if early_amplitude > 0:
                amplitude_decrease = (early_amplitude - late_amplitude) / early_amplitude
                amplitude_decrease = max(0.0, amplitude_decrease)
        
        return {
            'smoothness': smoothness,
            'rhythm_consistency': self._calculate_rhythm_consistency(cycles),
            'stops_detected': stops_ratio,
            'amplitude_decrease': amplitude_decrease
        }
    
    def _classify_performance(self, analysis: Dict) -> int:
        """
        Классификация качества выполнения пробы по шкале 0-4
        
        Критерии оценки:
        4 - Отличное выполнение: полная амплитуда, стабильный ритм, высокая скорость
        3 - Хорошее выполнение: небольшие отклонения
        2 - Удовлетворительное выполнение: умеренные нарушения
        1 - Плохое выполнение: выраженные нарушения
        0 - Невозможно оценить или критичные нарушения
        """
        if not analysis:
            return 0
        
        # Критерии для оценки
        criteria = {
            'sufficient_amplitude': analysis.get('mean_amplitude', 0) >= self.MIN_AMPLITUDE_THRESHOLD,
            'sufficient_cycles': analysis.get('num_cycles', 0) >= self.MIN_CYCLES_THRESHOLD,
            'good_rhythm': analysis.get('rhythm_consistency', 0) >= self.RHYTHM_CONSISTENCY_THRESHOLD,
            'good_speed': analysis.get('speed_consistency', 0) >= self.SPEED_CONSISTENCY_THRESHOLD,
            'good_amplitude_consistency': analysis.get('amplitude_consistency', 0) >= self.AMPLITUDE_CONSISTENCY_THRESHOLD,
            'smooth_movement': analysis.get('smoothness', 0) >= 0.6,
            'few_stops': analysis.get('stops_detected', 1) <= 0.3,
            'minimal_amplitude_decrease': analysis.get('amplitude_decrease', 1) <= 0.3
        }
        
        # Подсчет выполненных критериев
        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        # Определение оценки
        if passed_criteria >= 7:  # 7-8 критериев
            return 4
        elif passed_criteria >= 5:  # 5-6 критериев
            return 3
        elif passed_criteria >= 3:  # 3-4 критерия
            return 2
        elif passed_criteria >= 1:  # 1-2 критерия
            return 1
        else:  # 0 критериев
            return 0
    
    def _clinical_score(self, technical_score: int) -> int:
        """
        Инвертированная клиническая шкала: 4 (хуже) .. 0 (лучше)
        Согласно описанию: 4 - наибольший уровень неврологической декомпенсации
        """
        return max(0, min(4, 4 - technical_score))
    
    def _analyze_temporal_patterns(self, angles: np.ndarray, timestamps: np.ndarray) -> Dict:
        """Анализ временных паттернов нарушений"""
        if len(angles) < 10:
            return {
                'early_violations': 0.0,
                'mid_violations': 0.0,
                'late_violations': 0.0
            }
        
        # Разделение на трети времени
        total_duration = timestamps[-1] - timestamps[0]
        third_duration = total_duration / 3
        
        early_end = timestamps[0] + third_duration
        mid_end = timestamps[0] + 2 * third_duration
        
        # Анализ качества движений в каждой трети
        early_indices = timestamps <= early_end
        mid_indices = (timestamps > early_end) & (timestamps <= mid_end)
        late_indices = timestamps > mid_end
        
        violations = {'early': 0.0, 'mid': 0.0, 'late': 0.0}
        
        for period, indices in [('early', early_indices), ('mid', mid_indices), ('late', late_indices)]:
            if np.any(indices):
                period_angles = angles[indices]
                period_timestamps = timestamps[indices]
                
                # Анализ качества движений в периоде
                period_cycles = self._detect_rotation_cycles(period_angles, period_timestamps)
                period_quality = self._detect_movement_quality(period_angles, period_timestamps)
                
                # Оценка нарушений в периоде
                violations[period] = (
                    (1.0 - period_quality['smoothness']) +
                    (1.0 - period_quality['rhythm_consistency']) +
                    period_quality['stops_detected'] +
                    period_quality['amplitude_decrease']
                ) / 4.0
        
        return violations
    
    def analyze(self) -> Dict:
        """Основной метод анализа данных"""
        if self.data.empty:
            return self.hand_results
        
        # Анализ по группам рук
        if 'hand' in self.data.columns:
            groups = self.data.groupby('hand')
        else:
            groups = [(None, self.data)]
        
        for hand_label, df in groups:
            try:
                if hand_label == 'unknown':
                    continue
                
                # Получение данных
                angles = df['angle'].values
                timestamps = df['timestamp'].values
                
                if len(angles) < 10:
                    continue
                
                # Определение руки
                hand = self._hand_from_label(hand_label) if hand_label is not None else Hand.RIGHT
                
                # Детекция циклов
                cycles = self._detect_rotation_cycles(angles, timestamps)
                
                # Анализ качества движений
                movement_quality = self._detect_movement_quality(angles, timestamps)
                
                # Расчет метрик
                mean_amplitude = np.mean([c['amplitude'] for c in cycles]) if cycles else 0
                mean_duration = np.mean([c['duration'] for c in cycles]) if cycles else 0
                frequency = 1.0 / mean_duration if mean_duration > 0 else 0
                
                rhythm_consistency = self._calculate_rhythm_consistency(cycles)
                speed_consistency = self._calculate_speed_consistency(angles, timestamps)
                amplitude_consistency = self._calculate_amplitude_consistency(cycles)
                
                # Временные паттерны
                temporal_patterns = self._analyze_temporal_patterns(angles, timestamps)
                
                # Объединение всех метрик
                analysis = {
                    'num_cycles': len(cycles),
                    'mean_amplitude': mean_amplitude,
                    'mean_duration': mean_duration,
                    'frequency': frequency,
                    'rhythm_consistency': rhythm_consistency,
                    'speed_consistency': speed_consistency,
                    'amplitude_consistency': amplitude_consistency,
                    **movement_quality,
                    **temporal_patterns
                }
                
                # Классификация
                technical_score = self._classify_performance(analysis)
                clinical_score = self._clinical_score(technical_score)
                
                # Сохранение результатов
                self.hand_results[hand] = {
                    'technical_score': technical_score,
                    'clinical_score': clinical_score,
                    'analysis': analysis,
                    'cycles': cycles,
                    'angles': angles.tolist(),
                    'timestamps': timestamps.tolist()
                }
                
            except Exception as e:
                print(f"Ошибка при анализе ({hand_label}): {str(e)}")
        
        return self.hand_results
    
    def generate_report(self) -> str:
        """Генерация текстового отчета"""
        if not any(self.hand_results.values()):
            return "Нет данных для анализа"
        
        report = []
        report.append("=== АНАЛИЗ ПРОНАЦИИ-СУПИНАЦИИ КИСТЕЙ ===\n")
        
        for hand, result in self.hand_results.items():
            if result is None:
                continue
            
            report.append(f"--- {hand.value.upper()} РУКА ---")
            report.append(f"Техническая оценка: {result['technical_score']}/4")
            report.append(f"Клиническая оценка: {result['clinical_score']}/4")
            
            analysis = result['analysis']
            
            # Интерпретация оценки
            if result['clinical_score'] == 0:
                report.append("Интерпретация: Отличное выполнение - полная амплитуда, стабильный ритм")
            elif result['clinical_score'] == 1:
                report.append("Интерпретация: Хорошее выполнение - незначительные отклонения")
            elif result['clinical_score'] == 2:
                report.append("Интерпретация: Удовлетворительное выполнение - умеренные нарушения")
            elif result['clinical_score'] == 3:
                report.append("Интерпретация: Плохое выполнение - выраженные нарушения")
            else:
                report.append("Интерпретация: Критичные нарушения - невозможность выполнения")
            
            report.append("\nДетальные метрики:")
            report.append(f"- Количество циклов: {analysis.get('num_cycles', 0)}")
            report.append(f"- Средняя амплитуда: {analysis.get('mean_amplitude', 0):.1f}°")
            report.append(f"- Частота движений: {analysis.get('frequency', 0):.2f} Гц")
            report.append(f"- Ритмичность: {analysis.get('rhythm_consistency', 0)*100:.1f}%")
            report.append(f"- Стабильность скорости: {analysis.get('speed_consistency', 0)*100:.1f}%")
            report.append(f"- Плавность движений: {analysis.get('smoothness', 0)*100:.1f}%")
            report.append(f"- Остановки: {analysis.get('stops_detected', 0)*100:.1f}%")
            report.append(f"- Уменьшение амплитуды: {analysis.get('amplitude_decrease', 0)*100:.1f}%")
            
            report.append("\nВременные паттерны нарушений:")
            report.append(f"- Начало теста: {analysis.get('early_violations', 0)*100:.1f}%")
            report.append(f"- Середина теста: {analysis.get('mid_violations', 0)*100:.1f}%")
            report.append(f"- Конец теста: {analysis.get('late_violations', 0)*100:.1f}%")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_to_csv(self, output_path: str):
        """Сохранение результатов анализа в CSV"""
        if not any(self.hand_results.values()):
            return
        
        results = []
        for hand, result in self.hand_results.items():
            if result is not None:
                analysis = result['analysis']
                row = {
                    'hand': hand.value,
                    'technical_score': result['technical_score'],
                    'clinical_score': result['clinical_score'],
                    'num_cycles': analysis.get('num_cycles', 0),
                    'mean_amplitude': analysis.get('mean_amplitude', 0),
                    'frequency': analysis.get('frequency', 0),
                    'rhythm_consistency': analysis.get('rhythm_consistency', 0),
                    'speed_consistency': analysis.get('speed_consistency', 0),
                    'amplitude_consistency': analysis.get('amplitude_consistency', 0),
                    'smoothness': analysis.get('smoothness', 0),
                    'stops_detected': analysis.get('stops_detected', 0),
                    'amplitude_decrease': analysis.get('amplitude_decrease', 0),
                    'early_violations': analysis.get('early_violations', 0),
                    'mid_violations': analysis.get('mid_violations', 0),
                    'late_violations': analysis.get('late_violations', 0)
                }
                results.append(row)
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Результаты сохранены в {output_path}")
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """Визуализация результатов анализа"""
        if not any(self.hand_results.values()):
            print("Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ пронации-супинации кистей', fontsize=16)
        
        for idx, (hand, result) in enumerate(self.hand_results.items()):
            if result is None:
                continue
            
            angles = result['angles']
            timestamps = result['timestamps']
            
            # График углов во времени
            axes[0, idx].plot(timestamps, angles, 'b-', linewidth=2, label=f'{hand.value} рука')
            axes[0, idx].set_title(f'Углы пронации-супинации - {hand.value} рука')
            axes[0, idx].set_xlabel('Время (с)')
            axes[0, idx].set_ylabel('Угол (градусы)')
            axes[0, idx].grid(True, alpha=0.3)
            axes[0, idx].legend()
            
            # График циклов
            cycles = result['cycles']
            if cycles:
                cycle_durations = [c['duration'] for c in cycles]
                cycle_amplitudes = [c['amplitude'] for c in cycles]
                
                axes[1, idx].scatter(cycle_durations, cycle_amplitudes, 
                                   c='red', s=50, alpha=0.7, label='Циклы')
                axes[1, idx].set_title(f'Характеристики циклов - {hand.value} рука')
                axes[1, idx].set_xlabel('Длительность цикла (с)')
                axes[1, idx].set_ylabel('Амплитуда (градусы)')
                axes[1, idx].grid(True, alpha=0.3)
                axes[1, idx].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        
        plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ пронации-супинации кистей')
    parser.add_argument('--input', required=True, help='Путь к CSV с данными трекинга')
    parser.add_argument('--fps', type=float, default=30, help='Кадровая частота исходного видео')
    parser.add_argument('--output', default="wrist_analysis_results.csv", help='Путь для сохранения результатов')
    parser.add_argument('--plot', action='store_true', help='Показать графики анализа')
    parser.add_argument('--plot-save', help='Путь для сохранения графиков')
    
    args = parser.parse_args()
    
    analyzer = PronationSupinationAnalyzer(args.input, args.fps)
    analyzer.analyze()
    
    # Вывод отчета
    report = analyzer.generate_report()
    print(report)
    
    # Сохранение результатов
    analyzer.save_to_csv(args.output)
    
    # Визуализация
    if args.plot or args.plot_save:
        analyzer.plot_analysis(args.plot_save)

if __name__ == "__main__":
    main()
