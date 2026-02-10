# الخلية 1: التثبيت
!apt-get install -y swig
!pip install gymnasium[box2d] stable-baselines3

# الخلية 2: الاستيراد
import gymnasium as gym
from stable_baselines3 import PPO

# الخلية 3: إعداد البيئة
# ملاحظة: continuous=True يعني أن التحكم في المقود ودواسة الوقود مستمر وليس متقطعاً.
env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")

# الخلية 4: اختيار سياسة الشبكة (مساحة عمل الطالب)
# ==============================================================================
# التحدي: المدخلات هنا عبارة عن صور (Pixels)، لذا يجب استخدام CnnPolicy.
# المطلوب: ضبط معاملات خوارزمية PPO للتعامل مع الصور المعقدة.
# ==============================================================================

model = PPO(
    "CnnPolicy",  # استخدام الشبكات العصبية الالتفافية (ضروري للصور)
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,    # عدد الخطوات قبل التحديث
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01    # معامل الانتروبي لتشجيع الاستكشاف
)

# الخلية 5: التدريب المكثف
# تحذير: تدريب التعامل مع الصور يتطلب وقتاً وموارد حسابية (GPU يفضل تفعيله في Colab).
print("بدء تدريب السائق الآلي...")
model.learn(total_timesteps=50000) 

# الخلية 6: التقييم
# الحفظ لاستخدامه لاحقاً
model.save("eco_racing_model")
print("تم حفظ النموذج.")
