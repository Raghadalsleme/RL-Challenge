#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========================================================================
# تحدي التعلم الذاتي (للمتخصصين)
# المستوى الثالث: سباق توفير الوقود (Eco-Racing)
# الصعوبة: صعب
# 
# اسم الملف: eco_racing_challenge.py
# تاريخ الإنشاء: 2025
# ========================================================================

"""
 وصف التحدي:
--------------
سيارة سباق يجب عليها إكمال مسار السباق بأسرع وقت ممكن
مع توفير استهلاك الوقود. التحدي يجمع بين السرعة والكفاءة.

البيئة: CarRacing-v2 من Gymnasium
الهدف: قيادة السيارة على المسار الأخضر وتجنب الخروج عنه

 القوانين والقيود:
-------------------
1. يجب استخدام خوارزمية Q-Learning فقط
2. لا يسمح باستخدام Deep Learning أو Neural Networks
3. الصورة 96×96×3 يجب تبسيطها لحالات منفصلة
4. الإجراءات المسموحة (مستمرة):
   - التوجيه: -1 (يسار) إلى +1 (يمين)
   - التسارع: 0 إلى +1
   - الفرامل: 0 إلى +1
5. النجاح = إكمال المسار بمكافأة عالية

 معايير التقييم:
------------------
- البقاء على المسار: +1000/دورة كاملة
- الخروج عن المسار: -0.1 لكل إطار
- السرعة العالية: مكافآت إضافية
- المجموع النهائي: متوسط آخر 100 حلقة

 تنبيهات هامة:
-----------------
- هذا التحدي صعب جداً مع Q-Learning التقليدي
- يحتاج إبداع في تبسيط المشاهدات البصرية
- التدريب قد يستغرق وقتاً طويلاً جداً
- يُنصح بتقسيم الإجراءات إلى خيارات منفصلة
"""

# ========================================================================
# 1️⃣ استيراد المكتبات المطلوبة
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import gymnasium as gym
from collections import defaultdict
import cv2
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 2️⃣ إعداد البيئة (لا تعدل هذا القسم!)
# ========================================================================

class EcoRacingChallenge:
    """
    بيئة تحدي Eco-Racing
     ممنوع التعديل على هذا الكلاس!
    
    التحدي:
    - تحويل صورة 96×96×3 إلى حالة منفصلة
    - تبسيط الإجراءات المستمرة إلى منفصلة
    - الموازنة بين السرعة وتوفير الوقود
    """
    
    def __init__(self):
        self.env = gym.make('CarRacing-v2', continuous=False)
        self.action_space_size = 5  # إجراءات منفصلة مبسطة
        
    def simplify_observation(self, observation):
        """
        تبسيط المشاهدة البصرية إلى حالة منفصلة
        
        الاستراتيجية:
        - تحويل الصورة لـ grayscale
        - تقسيم الصورة إلى شبكة (grid)
        - اكتشاف المسار الأخضر في كل خلية
        - استخراج موقع السيارة واتجاهها
        """
        # تحويل لـ grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        # تقليل الحجم
        small = cv2.resize(gray, (12, 12))
        
        # تقسيم إلى 4 مستويات
        discretized = (small / 64).astype(int)
        discretized = np.clip(discretized, 0, 3)
        
        # اكتشاف المسار (اللون الأخضر)
        green_channel = observation[:, :, 1]
        track_indicator = (green_channel > 100).astype(int)
        track_sum = track_indicator.sum()
        
        # تبسيط إلى 5 مستويات
        on_track = min(4, track_sum // 1000)
        
        # استخراج الاتجاه (تقريبي من الجزء السفلي من الصورة)
        bottom_section = observation[60:80, :, :]
        left_green = bottom_section[:, :32, 1].mean()
        center_green = bottom_section[:, 32:64, 1].mean()
        right_green = bottom_section[:, 64:, 1].mean()
        
        if center_green > max(left_green, right_green):
            direction = 1  # في المنتصف
        elif left_green > right_green:
            direction = 0  # يسار
        else:
            direction = 2  # يمين
        
        return (on_track, direction, tuple(discretized.flatten()[:20]))
    
    def reset(self):
        """إعادة تعيين البيئة"""
        observation, _ = self.env.reset()
        return self.simplify_observation(observation)
    
    def step(self, action):
        """
        تنفيذ خطوة في البيئة
        
        تحويل الإجراءات المنفصلة إلى مستمرة:
        0: لا شيء
        1: يسار
        2: يمين
        3: تسارع
        4: فرامل
        
        المكافآت (لا يمكن تعديلها):
        - البقاء على المسار: إيجابي
        - الخروج عن المسار: سلبي
        - السرعة: مكافآت إضافية
        """
        # تحويل الإجراء المنفصل إلى مستمر
        action_map = {
            0: [0, 0, 0],      # لا شيء
            1: [-1, 0, 0],     # يسار
            2: [1, 0, 0],      # يمين
            3: [0, 1, 0],      # تسارع
            4: [0, 0, 0.8],    # فرامل
        }
        
        continuous_action = action_map.get(action, [0, 0, 0])
        
        observation, reward, terminated, truncated, info = self.env.step(
            continuous_action
        )
        done = terminated or truncated
        
        return self.simplify_observation(observation), reward, done, info
    
    def render(self):
        """عرض البيئة"""
        return self.env.render()
    
    def close(self):
        """إغلاق البيئة"""
        self.env.close()


# ========================================================================
# 3️⃣ خوارزمية Q-Learning (يمكنك التعديل هنا!)
# ========================================================================

class QLearningAgent:
    """
    وكيل Q-Learning لتحدي Eco-Racing
    
     يمكنك تعديل:
    - قيم المعاملات
    - استراتيجية الاستكشاف
    - طريقة معالجة الحالات المعقدة
    
     لا يمكنك:
    - استخدام Neural Networks
    - تغيير الخوارزمية الأساسية
    
  
    """
    
    def __init__(self, 
                 n_actions=5,
                 learning_rate=0.1,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.9995):
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # جدول Q مع قيم افتراضية
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # إحصائيات
        self.training_episodes = 0
    
    def get_action(self, state, training=True):
        """اختيار إجراء باستخدام epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """تحديث جدول Q"""
        current_q = self.q_table[state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        target_q = reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """تقليل epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_episodes += 1


# ========================================================================
# 4️⃣ دالة التدريب
# ========================================================================

def train_eco_racing(agent, env, n_episodes=1000, max_steps=1000, verbose=True):
    """
    تدريب الوكيل على تحدي Eco-Racing
    
    
    """
    
    episode_rewards = []
    episode_lengths = []
    
    print("  بدء التدريب على Eco-Racing...")
    print("=" * 70)
    print("   Q-Learning التقليدي قد لا يكون كافياً")
    print("   التدريب قد يستغرق وقتاً طويلاً جداً (ساعات)")
    print("=" * 70)
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            best_reward = max(episode_rewards[-50:])
            
            print(f"الحلقة {episode + 1:4d} | "
                  f"متوسط المكافأة: {avg_reward:8.2f} | "
                  f"أفضل: {best_reward:8.2f} | "
                  f"الطول: {avg_length:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("=" * 70)
    print(" اكتمل التدريب!")
    
    return episode_rewards, episode_lengths


# ========================================================================
# 5️⃣ دوال التصور والتقييم
# ========================================================================

def plot_training_results(episode_rewards, episode_lengths):
    """رسم نتائج التدريب"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(' نتائج التدريب - تحدي Eco-Racing', 
                 fontsize=16, weight='bold')
    
    # منحنى المكافآت
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, color='blue')
    
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), 
                moving_avg, color='red', linewidth=2)
    
    ax1.set_xlabel('رقم الحلقة')
    ax1.set_ylabel('المكافأة الكلية')
    ax1.set_title('منحنى التعلم - المكافآت')
    ax1.grid(True, alpha=0.3)
    
    # أطوال الحلقات
    ax2 = axes[0, 1]
    ax2.plot(episode_lengths, alpha=0.3, color='green')
    
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, 
                                np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), 
                moving_avg, color='orange', linewidth=2)
    
    ax2.set_xlabel('رقم الحلقة')
    ax2.set_ylabel('عدد الخطوات')
    ax2.set_title('طول الحلقات')
    ax2.grid(True, alpha=0.3)
    
    # توزيع المكافآت
    ax3 = axes[1, 0]
    last_100 = episode_rewards[-100:]
    ax3.hist(last_100, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(last_100), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('المكافأة')
    ax3.set_ylabel('التكرار')
    ax3.set_title('توزيع المكافآت (آخر 100 حلقة)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # أفضل المكافآت
    ax4 = axes[1, 1]
    best_rewards = []
    for i in range(50, len(episode_rewards), 10):
        best_rewards.append(max(episode_rewards[i-50:i]))
    
    ax4.plot(range(50, len(episode_rewards), 10), best_rewards, 
            color='gold', linewidth=2, marker='o', markersize=3)
    ax4.set_xlabel('رقم الحلقة')
    ax4.set_ylabel('أفضل مكافأة')
    ax4.set_title('أفضل أداء (آخر 50 حلقة)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_agent(agent, env, n_episodes=20):
    """تقييم الوكيل المدرب"""
    
    print("\n" + "=" * 70)
    print(" تقييم الأداء النهائي...")
    print("=" * 70)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action = agent.get_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
    
    stats = {
        'متوسط_المكافأة': np.mean(episode_rewards),
        'انحراف_معياري_المكافأة': np.std(episode_rewards),
        'أفضل_مكافأة': np.max(episode_rewards),
        'أسوأ_مكافأة': np.min(episode_rewards),
        'متوسط_الخطوات': np.mean(episode_lengths),
        'المجموع_النهائي': np.sum(episode_rewards)
    }
    
    print(f"\n النتائج على {n_episodes} حلقة:")
    print(f"   • متوسط المكافأة: {stats['متوسط_المكافأة']:.2f} ± {stats['انحراف_معياري_المكافأة']:.2f}")
    print(f"   • أفضل مكافأة: {stats['أفضل_مكافأة']:.2f}")
    print(f"   • أسوأ مكافأة: {stats['أسوأ_مكافأة']:.2f}")
    print(f"   • متوسط عدد الخطوات: {stats['متوسط_الخطوات']:.1f}")
    print(f"\n المجموع النهائي للنقاط: {stats['المجموع_النهائي']:.0f}")
    
    print("\n ملاحظة:")
    print("   هذا التحدي صعب جداً مع Q-Learning التقليدي")
    print("   المكافآت السلبية متوقعة - الهدف هو التحسن التدريجي")
    
    print("=" * 70)
    
    return stats


# ========================================================================
# 6️⃣ التشغيل الرئيسي
# ========================================================================

def main():
    """البرنامج الرئيسي للتحدي"""
    
    print("\n" + "=" * 70)
    print("🏎️  تحدي Eco-Racing - المستوى الثالث (صعب)")
    print("=" * 70)
    
    # إنشاء البيئة
    env = EcoRacingChallenge()
    
    # إنشاء الوكيل
    agent = QLearningAgent(
        n_actions=5,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995
    )
    
    print("\n⚙️  معاملات التعلم المستخدمة:")
    print(f"   • معدل التعلم (α): {agent.learning_rate}")
    print(f"   • معامل الخصم (γ): {agent.discount_factor}")
    print(f"   • Epsilon النهائي: {agent.epsilon_end}")
    print(f"   • معدل تناقص Epsilon: {agent.epsilon_decay}")
    
    print("\n  تحذير مهم:")
    print("   هذا التحدي معقد جداً لـ Q-Learning التقليدي")
    print("   النتائج قد تكون محدودة مقارنة بـ Deep RL")
    print("   الهدف: التعلم من الفشل والتحسن التدريجي")
    
    # التدريب
    episode_rewards, episode_lengths = train_eco_racing(
        agent, env, 
        n_episodes=500,
        max_steps=1000,
        verbose=True
    )
    
    # رسم النتائج
    plot_training_results(episode_rewards, episode_lengths)
    
    # التقييم النهائي
    final_stats = evaluate_agent(agent, env, n_episodes=20)
    
    # إغلاق البيئة
    env.close()
    
    return agent, env, final_stats


# ========================================================================
#  تشغيل التحدي
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  تثبيت المكتبات المطلوبة:")
    print("   pip install gymnasium opencv-python")
    print("   pip install gymnasium[box2d]")
    print("=" * 70)
    
    agent, env, stats = main()
    
    print("\n انتهى التحدي!")
    print("\n نصائح:")
    print("   - هذا التحدي يحتاج Deep RL للأداء الجيد")
    print("   - Q-Learning التقليدي محدود هنا")
    print("   - الهدف: فهم حدود الطرق التقليدية")
    print("   - جرب تحسين تبسيط الحالات")
