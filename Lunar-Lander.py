#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========================================================================
# تحدي التعلم الذاتي (للمتخصصين)
# المستوى الثاني: هبوط القمر (Lunar Lander)
# الصعوبة: متوسط
# 
# اسم الملف: lunar_lander_challenge.py
# تاريخ الإنشاء: 2025
# ========================================================================

"""
 وصف التحدي:
--------------
مركبة فضائية تحاول الهبوط بأمان على سطح القمر بين العلمين.
يجب التحكم في المحركات الرئيسية والجانبية للهبوط الآمن.

التحديات:
- إدارة الوقود المحدود
- التحكم في السرعة والدوران
- الهبوط في المنطقة الآمنة
- عدم الاصطدام بقوة

 القوانين والقيود:
-------------------
1. يجب استخدام خوارزمية Q-Learning فقط
2. لا يسمح باستخدام Deep Learning أو Neural Networks
3. يجب تقسيم الحالات (State Discretization) - 8 أبعاد مستمرة
4. الإجراءات المسموحة: 
   - 0: لا شيء
   - 1: محرك أيسر
   - 2: محرك رئيسي
   - 3: محرك أيمن
5. النجاح = الهبوط الآمن بمكافأة إيجابية

 معايير التقييم:
------------------
- مكافأة الهبوط الآمن: +100 إلى +140
- استخدام الوقود: -0.3 لكل محرك
- تحطم المركبة: -100
- المجموع النهائي: متوسط آخر 100 حلقة

 تنبيهات هامة:
-----------------
- لا تقم بتعديل البيئة أو قوانين المكافآت
- يمكنك فقط تعديل معاملات التعلم وطريقة التقسيم
- التحدي أصعب من Mountain Car - يحتاج استراتيجيات متقدمة
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

class LunarLanderChallenge:
    """
    بيئة تحدي Lunar Lander
     ممنوع التعديل على هذا الكلاس!
    
    الحالة (8 أبعاد):
    - x: الموقع الأفقي
    - y: الموقع العمودي
    - vx: السرعة الأفقية
    - vy: السرعة العمودية
    - angle: زاوية الدوران
    - angular_velocity: سرعة الدوران
    - leg1_contact: ملامسة الساق اليسرى للأرض
    - leg2_contact: ملامسة الساق اليمنى للأرض
    """
    
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.state_bins = None
        
    def setup_discretization(self, bins_per_dimension=10):
        """
        إعداد تقسيم الحالات المستمرة
        
        المعاملات:
        -----------
        bins_per_dimension: عدد الأقسام لكل بُعد (افتراضي: 10)
        
        ملاحظة: زيادة العدد = دقة أعلى لكن وقت تدريب أطول
        """
        # حدود تقريبية لكل بُعد من الحالة
        self.state_bounds = [
            (-1.5, 1.5),    # x position
            (-1.5, 1.5),    # y position
            (-2.5, 2.5),    # x velocity
            (-2.5, 2.5),    # y velocity
            (-3.14, 3.14),  # angle
            (-5.0, 5.0),    # angular velocity
            (0, 1),         # leg 1 contact (binary)
            (0, 1)          # leg 2 contact (binary)
        ]
        
        self.state_bins = []
        for i, (low, high) in enumerate(self.state_bounds):
            if i >= 6:  # الأبعاد الثنائية (الساقين)
                self.state_bins.append(np.array([0, 1]))
            else:
                self.state_bins.append(
                    np.linspace(low, high, bins_per_dimension)
                )
    
    def discretize_state(self, state):
        """تحويل الحالة المستمرة إلى منفصلة"""
        discrete_state = []
        
        for i, (s, bins) in enumerate(zip(state, self.state_bins)):
            # تقييد القيمة ضمن الحدود
            s_clipped = np.clip(s, self.state_bounds[i][0], 
                               self.state_bounds[i][1])
            # تحويل لفهرس منفصل
            idx = np.digitize(s_clipped, bins)
            discrete_state.append(idx)
        
        return tuple(discrete_state)
    
    def reset(self):
        """إعادة تعيين البيئة"""
        state, _ = self.env.reset()
        return self.discretize_state(state)
    
    def step(self, action):
        """
        تنفيذ خطوة في البيئة
        
        المكافآت (لا يمكن تعديلها):
        - الهبوط الآمن: +100 إلى +140
        - التحطم: -100
        - استخدام المحركات: -0.3 لكل إطار
        - الحركة نحو الهبوط: مكافآت تدريجية
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
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
    وكيل Q-Learning لتحدي Lunar Lander
    
     يمكنك تعديل:
    - قيم المعاملات (learning_rate, discount_factor, etc.)
    - استراتيجية epsilon decay
    - طريقة اختيار الإجراء
    
     لا يمكنك:
    - استخدام Neural Networks
    - تغيير الخوارزمية الأساسية
    """
    
    def __init__(self, 
                 n_actions=4,
                 learning_rate=0.1,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995):
        """
        المعاملات القابلة للتعديل:
        ---------------------------
        learning_rate: معدل التعلم (alpha) - جرب قيم بين 0.01 و 0.3
        discount_factor: معامل الخصم (gamma) - جرب قيم بين 0.95 و 0.999
        epsilon_start: قيمة epsilon الابتدائية
        epsilon_end: قيمة epsilon النهائية
        epsilon_decay: معدل تناقص epsilon
        """
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # جدول Q باستخدام defaultdict
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # إحصائيات
        self.training_episodes = 0
    
    def get_action(self, state, training=True):
        """
        اختيار إجراء باستخدام epsilon-greedy
        
        يمكنك تعديل هذه الدالة لتحسين الأداء!
        مثلاً: استخدام استراتيجيات استكشاف متقدمة
        """
        if training and np.random.random() < self.epsilon:
            # استكشاف: اختيار عشوائي
            return np.random.randint(0, self.n_actions)
        else:
            # استغلال: اختيار أفضل إجراء
            q_values = self.q_table[state]
            # في حالة تساوي القيم، اختر عشوائياً
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """
        تحديث جدول Q باستخدام معادلة Q-Learning
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            # إذا انتهت الحلقة
            max_next_q = 0
        else:
            # أقصى قيمة Q للحالة التالية
            max_next_q = np.max(self.q_table[next_state])
        
        # حساب القيمة المستهدفة
        target_q = reward + self.discount_factor * max_next_q
        
        # تحديث Q
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """تقليل epsilon بعد كل حلقة"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_episodes += 1


# ========================================================================
# 4️⃣ دالة التدريب
# ========================================================================

def train_lunar_lander(agent, env, n_episodes=2000, max_steps=1000, verbose=True):
    """
    تدريب الوكيل على تحدي Lunar Lander
    
    المعاملات:
    -----------
    agent: وكيل Q-Learning
    env: بيئة التحدي
    n_episodes: عدد الحلقات التدريبية (يُنصح بـ 2000+)
    max_steps: الحد الأقصى للخطوات في كل حلقة
    verbose: عرض التقدم
    
    المخرجات:
    ---------
    episode_rewards: قائمة بمكافآت كل حلقة
    episode_lengths: قائمة بأطوال كل حلقة
    success_count: عدد مرات الهبوط الناجح
    """
    
    episode_rewards = []
    episode_lengths = []
    success_episodes = []
    
    print(" بدء التدريب على Lunar Lander...")
    print("=" * 70)
    print("  تنبيه: التدريب قد يستغرق وقتاً طويلاً (10-30 دقيقة)")
    print("=" * 70)
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # اختيار وتنفيذ إجراء
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # تحديث Q-table
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # تقليل epsilon
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # تتبع الحلقات الناجحة
        if episode_reward >= 200:
            success_episodes.append(episode)
        
        # عرض التقدم
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success_rate = len([r for r in episode_rewards[-100:] if r >= 200])
            
            print(f"الحلقة {episode + 1:4d} | "
                  f"متوسط المكافأة: {avg_reward:8.2f} | "
                  f"متوسط الطول: {avg_length:6.1f} | "
                  f"نجاحات: {success_rate:2d}/100 | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("=" * 70)
    print("✅ اكتمل التدريب!")
    print(f"   إجمالي الهبوط الناجح: {len(success_episodes)}")
    
    return episode_rewards, episode_lengths, success_episodes


# ========================================================================
# 5️⃣ دوال التصور والتقييم
# ========================================================================

def plot_training_results(episode_rewards, episode_lengths):
    """رسم نتائج التدريب"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📊 نتائج التدريب - تحدي Lunar Lander', 
                 fontsize=16, weight='bold')
    
    # 1. منحنى المكافآت
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.2, color='blue', label='المكافأة')
    
    # خط النجاح
    ax1.axhline(y=200, color='green', linestyle='--', 
                linewidth=2, label='حد النجاح (200)')
    
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), 
                moving_avg, color='red', linewidth=2, 
                label=f'المتوسط المتحرك ({window})')
    
    ax1.set_xlabel('رقم الحلقة', fontsize=11)
    ax1.set_ylabel('المكافأة الكلية', fontsize=11)
    ax1.set_title('منحنى التعلم - المكافآت', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. أطوال الحلقات
    ax2 = axes[0, 1]
    ax2.plot(episode_lengths, alpha=0.2, color='green', label='الطول')
    
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, 
                                np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), 
                moving_avg, color='orange', linewidth=2, 
                label=f'المتوسط المتحرك ({window})')
    
    ax2.set_xlabel('رقم الحلقة', fontsize=11)
    ax2.set_ylabel('عدد الخطوات', fontsize=11)
    ax2.set_title('طول الحلقات', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. توزيع المكافآت (آخر 200 حلقة)
    ax3 = axes[1, 0]
    last_episodes = episode_rewards[-200:]
    ax3.hist(last_episodes, bins=40, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(last_episodes), color='red', linestyle='--', 
                linewidth=2, label=f'المتوسط: {np.mean(last_episodes):.1f}')
    ax3.axvline(200, color='green', linestyle='--', 
                linewidth=2, label='حد النجاح')
    ax3.set_xlabel('المكافأة', fontsize=11)
    ax3.set_ylabel('التكرار', fontsize=11)
    ax3.set_title('توزيع المكافآت (آخر 200 حلقة)', fontsize=12, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. معدل النجاح المتحرك
    ax4 = axes[1, 1]
    success_rates = []
    
    for i in range(100, len(episode_rewards), 10):
        recent = episode_rewards[i-100:i]
        success_rate = (np.array(recent) >= 200).mean() * 100
        success_rates.append(success_rate)
    
    ax4.plot(range(100, len(episode_rewards), 10), success_rates, 
            color='teal', linewidth=2, marker='o', markersize=3)
    ax4.axhline(y=90, color='gold', linestyle='--', 
                linewidth=2, alpha=0.5, label='هدف 90%')
    ax4.set_xlabel('رقم الحلقة', fontsize=11)
    ax4.set_ylabel('معدل النجاح (%)', fontsize=11)
    ax4.set_title('معدل النجاح (آخر 100 حلقة)', fontsize=12, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.show()


def evaluate_agent(agent, env, n_episodes=100):
    """
    تقييم الوكيل المدرب
    
    المخرجات:
    ---------
    dict: إحصائيات الأداء
    """
    
    print("\n" + "=" * 70)
    print("📈 تقييم الأداء النهائي...")
    print("=" * 70)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    crash_count = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action = agent.get_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                if episode_reward >= 200:
                    success_count += 1
                elif episode_reward < -100:
                    crash_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
    
    # حساب الإحصائيات
    stats = {
        'متوسط_المكافأة': np.mean(episode_rewards),
        'انحراف_معياري_المكافأة': np.std(episode_rewards),
        'أفضل_مكافأة': np.max(episode_rewards),
        'أسوأ_مكافأة': np.min(episode_rewards),
        'متوسط_الخطوات': np.mean(episode_lengths),
        'معدل_النجاح_%': (success_count / n_episodes) * 100,
        'معدل_التحطم_%': (crash_count / n_episodes) * 100,
        'المجموع_النهائي': np.sum(episode_rewards)
    }
    
    # عرض النتائج
    print(f"\n النتائج على {n_episodes} حلقة:")
    print(f"   • متوسط المكافأة: {stats['متوسط_المكافأة']:.2f} ± {stats['انحراف_معياري_المكافأة']:.2f}")
    print(f"   • أفضل مكافأة: {stats['أفضل_مكافأة']:.2f}")
    print(f"   • أسوأ مكافأة: {stats['أسوأ_مكافأة']:.2f}")
    print(f"   • متوسط عدد الخطوات: {stats['متوسط_الخطوات']:.1f}")
    print(f"   • معدل الهبوط الناجح: {stats['معدل_النجاح_%']:.1f}%")
    print(f"   • معدل التحطم: {stats['معدل_التحطم_%']:.1f}%")
    print(f"\n المجموع النهائي للنقاط: {stats['المجموع_النهائي']:.0f}")
    
    # تقييم الأداء
    if stats['معدل_النجاح_%'] >= 90:
        print("⭐⭐⭐ أداء ممتاز! وصلت للمستوى المطلوب!")
    elif stats['معدل_النجاح_%'] >= 70:
        print("⭐⭐ أداء جيد جداً! قريب من الهدف!")
    elif stats['معدل_النجاح_%'] >= 50:
        print("⭐ أداء جيد! يمكن تحسينه أكثر")
    else:
        print("💡 يحتاج تحسين - جرب تعديل المعاملات أو زيادة التدريب")
    
    print("=" * 70)
    
    return stats


# ========================================================================
# 6️⃣ التشغيل الرئيسي
# ========================================================================

def main():
    """البرنامج الرئيسي للتحدي"""
    
    print("\n" + "=" * 70)
    print(" تحدي Lunar Lander - المستوى الثاني (متوسط)")
    print("=" * 70)
    
    # إنشاء البيئة
    env = LunarLanderChallenge()
    
    # إعداد تقسيم الحالات (يمكنك تعديل هذه القيم!)
    # تنبيه: زيادة bins_per_dimension يزيد الدقة لكن يبطئ التدريب
    env.setup_discretization(bins_per_dimension=10)
    
    # إنشاء الوكيل (يمكنك تعديل المعاملات!)
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.1,         # جرب: 0.05, 0.15, 0.2
        discount_factor=0.99,       # جرب: 0.95, 0.98, 0.999
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995        # جرب: 0.995, 0.999
    )
    
    print("\n⚙️  معاملات التعلم المستخدمة:")
    print(f"   • معدل التعلم (α): {agent.learning_rate}")
    print(f"   • معامل الخصم (γ): {agent.discount_factor}")
    print(f"   • Epsilon النهائي: {agent.epsilon_end}")
    print(f"   • معدل تناقص Epsilon: {agent.epsilon_decay}")
    print(f"   • تقسيم الحالات: 10^8 (8 أبعاد)")
    
    # التدريب
    print("\n نصيحة: Lunar Lander أصعب من Mountain Car")
    print("   قد تحتاج 2000+ حلقة للوصول لأداء جيد")
    
    episode_rewards, episode_lengths, success_episodes = train_lunar_lander(
        agent, env, 
        n_episodes=2000,  # يمكنك زيادة العدد لنتائج أفضل
        max_steps=1000,
        verbose=True
    )
    
    # رسم النتائج
    plot_training_results(episode_rewards, episode_lengths)
    
    # التقييم النهائي
    final_stats = evaluate_agent(agent, env, n_episodes=100)
    
    # إغلاق البيئة
    env.close()
    
    return agent, env, final_stats


# ========================================================================
#  تشغيل التحدي
# ========================================================================

if __name__ == "__main__":
    agent, env, stats = main()
    
    print("\n انتهى التحدي!")
    print(" نصائح للتحسين:")
    print("   - جرب زيادة عدد حلقات التدريب (3000-5000)")
    print("   - اضبط معدل التعلم (learning_rate)")
    print("   - جرب قيم مختلفة لـ epsilon_decay")
    print("   - يمكنك تقليل bins_per_dimension للتدريب الأسرع")
    print("\n لا تنسَ توثيق استراتيجيتك والتغييرات!")
