#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========================================================================
# تحدي التعلم الذاتي (للمتخصصين)
# المستوى الرابع: الروبوت الماشي (Bipedal Walker)
# الصعوبة: خبير
# 
# اسم الملف: bipedal_walker_challenge.py
# تاريخ الإنشاء: 2025
# ========================================================================

"""
 وصف التحدي:
--------------
روبوت ذو ساقين يحاول المشي عبر تضاريس وعرة.
يجب التحكم في 4 محركات (مفاصل) لتحقيق المشي المستقر.

البيئة: BipedalWalker-v3
الهدف: المشي لأبعد مسافة ممكنة دون السقوط

 القوانين والقيود:
-------------------
1. يجب استخدام خوارزمية Q-Learning فقط
2. لا يسمح باستخدام Deep Learning أو Neural Networks
3. الحالة: 24 بُعد مستمر (زوايا، سرعات، ملامسة الأرض...)
4. الإجراءات: 4 قيم مستمرة بين -1 و +1 لكل مفصل
5. النجاح = مكافأة > 300 (المشي الناجح)

الحالة (24 بُعد):
- سرعة الهيكل الأفقية والعمودية
- سرعة الدوران الزاوية
- زوايا المفاصل (4 مفاصل)
- سرعات المفاصل الزاوية
- معلومات اللامسة للأرض (نقاط LIDAR - 10 نقاط)
- ملامسة القدمين للأرض

 معايير التقييم:
------------------
- المشي للأمام: +1 لكل إطار
- استخدام المحركات: -0.00035 لكل محرك
- السقوط: -100
- المكافأة المستهدفة: > 300
- المجموع النهائي: متوسط آخر 100 حلقة

 تنبيهات هامة:
-----------------
- هذا التحدي للخبراء فقط
- Q-Learning التقليدي غير مناسب تماماً
- الحالة والإجراءات عالية الأبعاد
- يحتاج استراتيجيات تبسيط إبداعية جداً

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
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 2️⃣ إعداد البيئة (لا تعدل هذا القسم!)
# ========================================================================

class BipedalWalkerChallenge:
    """
    بيئة تحدي Bipedal Walker
     ممنوع التعديل على هذا الكلاس!
    
    التحدي:
    - 24 بُعد مستمر للحالة
    - 4 إجراءات مستمرة
    - فيزياء معقدة
    - يحتاج توازن دقيق
    """
    
    def __init__(self):
        # استخدام الإصدار المنفصل للتبسيط
        self.env = gym.make('BipedalWalker-v3')
        
        # حدود تقريبية للحالات
        self.state_bounds = [
            (-5, 5),    # hull angle speed
            (-5, 5),    # hull angular velocity
        ] + [(-3, 3)] * 4  # joint angles
        
        self.n_actions_per_joint = 3  # 3 خيارات لكل مفصل
        self.total_actions = self.n_actions_per_joint ** 4  # 81 إجراء
        
    def discretize_state(self, state):
        """
        تبسيط الحالة عالية الأبعاد
        
        الاستراتيجية:
        - استخدام أهم 6 أبعاد فقط
        - تجاهل معلومات LIDAR (مبسطة جداً)
        - التركيز على الزوايا والسرعات الأساسية
        """
        # استخراج أهم المعلومات
        hull_angle = state[0]
        hull_angular_vel = state[1]
        hip1_angle = state[4]
        knee1_angle = state[5]
        hip2_angle = state[8]
        knee2_angle = state[9]
        
        # تقسيم كل بُعد إلى 5 فئات
        discretized = []
        values = [hull_angle, hull_angular_vel, hip1_angle, 
                 knee1_angle, hip2_angle, knee2_angle]
        
        for i, val in enumerate(values):
            if i < len(self.state_bounds):
                low, high = self.state_bounds[i]
            else:
                low, high = -3, 3
            
            # تقسيم إلى 5 فئات
            normalized = (val - low) / (high - low)
            normalized = np.clip(normalized, 0, 1)
            category = int(normalized * 4)
            discretized.append(category)
        
        return tuple(discretized)
    
    def discretize_action(self, action_idx):
        """
        تحويل فهرس الإجراء المنفصل إلى 4 قيم مستمرة
        
        كل مفصل له 3 خيارات: -1, 0, +1
        81 إجراء ممكن (3^4)
        """
        actions = []
        remaining = action_idx
        
        for _ in range(4):
            action_val = remaining % self.n_actions_per_joint
            remaining //= self.n_actions_per_joint
            
            # تحويل 0,1,2 إلى -1,0,+1
            if action_val == 0:
                actions.append(-1.0)
            elif action_val == 1:
                actions.append(0.0)
            else:
                actions.append(1.0)
        
        return np.array(actions)
    
    def reset(self):
        """إعادة تعيين البيئة"""
        state, _ = self.env.reset()
        return self.discretize_state(state)
    
    def step(self, action_idx):
        """
        تنفيذ خطوة في البيئة
        
        المكافآت (لا يمكن تعديلها):
        - التقدم للأمام: +1 لكل إطار
        - استخدام المحركات: -0.00035
        - السقوط: -100
        """
        continuous_action = self.discretize_action(action_idx)
        
        next_state, reward, terminated, truncated, info = self.env.step(
            continuous_action
        )
        done = terminated or truncated
        
        return self.discretize_state(next_state), reward, done, info
    
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
    وكيل Q-Learning لتحدي Bipedal Walker
    
     يمكنك تعديل:
    - المعاملات
    - استراتيجيات الاستكشاف
    - طريقة معالجة الحالات
    
     لا يمكنك:
    - استخدام Neural Networks
    
   
    """
    
    def __init__(self, 
                 n_actions=81,
                 learning_rate=0.1,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.9995):
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # جدول Q
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # إحصائيات
        self.training_episodes = 0
    
    def get_action(self, state, training=True):
        """اختيار إجراء"""
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

def train_bipedal_walker(agent, env, n_episodes=2000, max_steps=1600, verbose=True):
    """
    تدريب الوكيل على تحدي Bipedal Walker
    
     تحذير: التدريب طويل جداً والنتائج محدودة
    """
    
    episode_rewards = []
    episode_lengths = []
    
    print(" بدء التدريب على Bipedal Walker...")
    print("=" * 70)
    print("  تحذير حرج:")
    print("   هذا التحدي شبه مستحيل مع Q-Learning التقليدي!")
    print("   المكافآت السلبية متوقعة")
    print("   الهدف: فهم حدود الطرق الكلاسيكية")
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
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            best_reward = max(episode_rewards[-100:])
            
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
    fig.suptitle(' نتائج التدريب - تحدي Bipedal Walker', 
                 fontsize=16, weight='bold')
    
    # منحنى المكافآت
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, color='blue')
    ax1.axhline(y=300, color='green', linestyle='--', 
                linewidth=2, label='هدف النجاح (300)')
    
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), 
                moving_avg, color='red', linewidth=2)
    
    ax1.set_xlabel('رقم الحلقة')
    ax1.set_ylabel('المكافأة الكلية')
    ax1.set_title('منحنى التعلم - المكافآت')
    ax1.legend()
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
    ax2.set_title('طول الحلقات (أطول = أفضل)')
    ax2.grid(True, alpha=0.3)
    
    # توزيع المكافآت
    ax3 = axes[1, 0]
    last_200 = episode_rewards[-200:]
    ax3.hist(last_200, bins=40, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(last_200), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('المكافأة')
    ax3.set_ylabel('التكرار')
    ax3.set_title('توزيع المكافآت (آخر 200 حلقة)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # التحسن عبر الوقت
    ax4 = axes[1, 1]
    improvement = []
    for i in range(100, len(episode_rewards), 50):
        improvement.append(np.mean(episode_rewards[i-100:i]))
    
    ax4.plot(range(100, len(episode_rewards), 50), improvement, 
            color='teal', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('رقم الحلقة')
    ax4.set_ylabel('متوسط المكافأة')
    ax4.set_title('التحسن التدريجي')
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
        
        for step in range(1600):
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
    print(f"\n  المجموع النهائي للنقاط: {stats['المجموع_النهائي']:.0f}")
    
    print("\n الواقع:")
    if stats['متوسط_المكافأة'] > -50:
        print("   نتيجة مقبولة نظراً لصعوبة التحدي!")
    else:
        print("   النتائج محدودة - هذا متوقع مع Q-Learning")
    
    print("   هذا التحدي يحتاج Deep RL (PPO, SAC, TD3)")
    print("   Q-Learning التقليدي غير مناسب للتحكم المستمر المعقد")
    
    print("=" * 70)
    
    return stats


# ========================================================================
# 6️⃣ التشغيل الرئيسي
# ========================================================================

def main():
    """البرنامج الرئيسي للتحدي"""
    
    print("\n" + "=" * 70)
    print(" تحدي Bipedal Walker - المستوى الرابع (خبير)")
    print("=" * 70)
    
    # إنشاء البيئة
    env = BipedalWalkerChallenge()
    
    # إنشاء الوكيل
    agent = QLearningAgent(
        n_actions=81,
        learning_rate=0.2,
        discount_factor=0.98,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9995
    )
    
    print("\n⚙️  معاملات التعلم المستخدمة:")
    print(f"   • معدل التعلم (α): {agent.learning_rate}")
    print(f"   • معامل الخصم (γ): {agent.discount_factor}")
    print(f"   • Epsilon النهائي: {agent.epsilon_end}")
    print(f"   • عدد الإجراءات: {agent.n_actions}")
    
    print("\n  تحذير للخبراء:")
    print("   هذا التحدي مصمم لـ Deep Reinforcement Learning")
    print("   Q-Learning التقليدي لن يحقق نتائج جيدة")
    print("   الهدف التعليمي: فهم متى نحتاج Deep RL")
    
    # التدريب
    episode_rewards, episode_lengths = train_bipedal_walker(
        agent, env, 
        n_episodes=1000,
        max_steps=1600,
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
# 🚀 تشغيل التحدي
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  تثبيت المكتبات المطلوبة:")
    print("   pip install gymnasium")
    print("   pip install gymnasium[box2d]")
    print("=" * 70)
    
    agent, env, stats = main()
    
    print("\n انتهى التحدي!")
    print("\n دروس مستفادة:")
    print("   - Q-Learning التقليدي له حدود واضحة")
    print("   - الحالات والإجراءات عالية الأبعاد تحتاج Deep RL")
    print("   - الفيزياء المعقدة تحتاج تقريب دوال القيمة")
    print("   - PPO, SAC, TD3 هي الخوارزميات المناسبة هنا")
