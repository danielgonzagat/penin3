# ⚡ Quick Start Guide

Get the system running in 2 minutes!

---

## 🚀 First Time Setup

```bash
cd /root/intelligence_system

# 1. Verify everything is ready
python3 verify_setup.py

# 2. Start the system
./start.sh

# 3. Check it's running
./status.sh
```

---

## 📊 Monitor Progress

### View logs in real-time:
```bash
tail -f logs/intelligence.log
```

### Check current metrics:
```bash
./status.sh
```

### View database:
```bash
sqlite3 data/intelligence.db "SELECT cycle, ROUND(mnist_accuracy,1) as MNIST, ROUND(cartpole_avg_reward,1) as CartPole FROM cycles ORDER BY cycle DESC LIMIT 10"
```

---

## 🛑 Stop/Restart

### Stop system:
```bash
./stop.sh
```

### Restart system:
```bash
./stop.sh && sleep 2 && ./start.sh
```

---

## 🧪 Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_mnist.py -v
pytest tests/test_dqn.py -v
```

---

## 📈 What to Expect

**After 10 minutes:**
- MNIST: 85-90%
- CartPole: 30-60

**After 1 hour:**
- MNIST: 92-96%
- CartPole: 100-200

**After 6 hours:**
- MNIST: 96-98%
- CartPole: 250-400

---

## 🐛 Troubleshooting

### System won't start:
```bash
python3 verify_setup.py
```

### Check for errors:
```bash
sqlite3 data/intelligence.db "SELECT * FROM errors ORDER BY timestamp DESC LIMIT 5"
```

### Reset everything:
```bash
./stop.sh
rm -f data/intelligence.db
rm -f models/*.pth
./start.sh
```

---

## ✅ System is Ready When:

- ✅ `verify_setup.py` passes all checks
- ✅ `./status.sh` shows "RUNNING"
- ✅ Logs show "CYCLE" messages
- ✅ Database has entries

---

**That's it! System is running and learning!** 🎉
