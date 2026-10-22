"""Local-disk SQLite queue with immutable attempts and fenced leases."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time
import uuid

TERMINAL = {'succeeded', 'failed', 'cancelled', 'blocked'}
KINDS = {'train', 'checkpoint_wait', 'export', 'inference', 'score', 'merge', 'publish', 'average', 'eval'}

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, config TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT 'pending', held INTEGER NOT NULL DEFAULT 0,
              owner TEXT, lease REAL, attempt INTEGER NOT NULL DEFAULT 0,
              max_attempts INTEGER NOT NULL DEFAULT 3, result TEXT, created REAL NOT NULL, worker_identity TEXT);
            CREATE TABLE IF NOT EXISTS dependencies (
              job TEXT REFERENCES jobs(id), dependency TEXT REFERENCES jobs(id),
              policy TEXT CHECK(policy IN ('success','terminal')), PRIMARY KEY(job,dependency));
            CREATE TABLE IF NOT EXISTS attempts (
              token TEXT PRIMARY KEY, job TEXT REFERENCES jobs(id), number INTEGER,
              config TEXT NOT NULL, started REAL, finished REAL, state TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS resources (
              name TEXT PRIMARY KEY, token TEXT NOT NULL REFERENCES attempts(token));
            ''')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        return db

    @contextmanager
    def transaction(self):
        db = self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally: db.close()

    def add(self, identity, kind, config, dependencies=(), max_attempts=3, *, held=False):
        if not isinstance(held, bool):
            raise ValueError('held must be a boolean')
        if kind not in KINDS or max_attempts < 1:
            raise ValueError('Invalid job kind or retry bound')
        if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in identity):
            raise ValueError('Job ID must be a safe filename')
        with self.transaction() as db:
            db.execute('INSERT INTO jobs(id,kind,config,max_attempts,created,held) VALUES(?,?,?,?,?,?)',
                       (identity, kind, json.dumps(config, allow_nan=False), max_attempts, time.time(), int(held)))
            for dependency, policy in dependencies:
                if dependency == identity:
                    raise ValueError('Self-dependency')
                parent=db.execute('SELECT kind FROM jobs WHERE id=?',(dependency,)).fetchone()
                if kind=='train' and parent:
                    if parent['kind'] in ('merge','publish','average'):
                        raise ValueError('Training cannot depend on administrative completion')
                    if parent['kind'] in ('eval','inference','score') and policy!='terminal':
                        raise ValueError('Training handoffs require terminal evaluation dependencies')
                db.execute('INSERT INTO dependencies VALUES(?,?,?)', (identity, dependency, policy))

    def edit(self, identity, config):
        with self.transaction() as db:
            if db.execute("UPDATE jobs SET config=? WHERE id=? AND state='pending'",
                          (json.dumps(config, allow_nan=False), identity)).rowcount != 1:
                raise ValueError('Only pending jobs can be edited')

    def hold(self, identity, value):
        with self.transaction() as db:
            db.execute('UPDATE jobs SET held=? WHERE id=?', (int(value), identity))

    def claim(self, ttl=120, predicate=None):
        if ttl <= 0: raise ValueError('Positive lease duration required')
        with self.transaction() as db:
            # Propagate failed prerequisites; terminal handoffs remain runnable.
            changed = True
            while changed:
                changed = db.execute("""UPDATE jobs SET state='blocked' WHERE state='pending' AND id IN
                  (SELECT d.job FROM dependencies d JOIN jobs j ON j.id=d.dependency
                   WHERE d.policy='success' AND j.state IN ('failed','cancelled','blocked'))""").rowcount
            candidates = db.execute("""SELECT * FROM jobs j WHERE state='pending' AND held=0 AND NOT EXISTS
              (SELECT 1 FROM dependencies d JOIN jobs p ON p.id=d.dependency WHERE d.job=j.id AND
               ((d.policy='success' AND p.state!='succeeded') OR
                (d.policy='terminal' AND p.state NOT IN ('succeeded','failed','cancelled','blocked'))))
              ORDER BY created,id""").fetchall()
            for row in candidates:
                config = json.loads(row['config']); resources = sorted(set(config.get('resources', [])))
                if predicate is not None and not predicate(dict(row) | {'config': config}):continue
                if any(db.execute('SELECT 1 FROM resources WHERE name=?', (r,)).fetchone() for r in resources):continue
                token = uuid.uuid4().hex; now = time.time(); number = row['attempt'] + 1
                import os, psutil
                identity=json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time()})
                db.execute("UPDATE jobs SET state='running',owner=?,lease=?,attempt=?,worker_identity=? WHERE id=?",
                           (token, now + ttl, number, identity, row['id']))
                db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)',
                           (token, row['id'], number, row['config'], now, None, 'running', None))
                db.executemany('INSERT INTO resources VALUES(?,?)', [(r, token) for r in resources])
                return dict(row) | {'owner': token, 'attempt': number, 'config': config}
        return None

    def heartbeat(self, token, ttl=120):
        with self.transaction() as db:
            if db.execute("UPDATE jobs SET lease=? WHERE owner=? AND state='running'", (time.time()+ttl,token)).rowcount != 1:
                raise RuntimeError('Lease ownership lost')

    def finish(self, token, success, result, *, transient=False, oom=False):
        with self.transaction() as db:
            job = db.execute("SELECT * FROM jobs WHERE owner=? AND state='running'", (token,)).fetchone()
            if job is None: raise RuntimeError('Stale attempt cannot finish')
            state = 'succeeded' if success else 'failed'
            config = json.loads(job['config'])
            retry = not success and (transient or oom) and job['attempt'] < job['max_attempts']
            if oom:
                # Adjust only declared backend controls; never rewrite arbitrary command strings.
                controls = config.setdefault('controls', {})
                controls['max_connections'] = max(1, controls.get('max_connections', 4)//2)
                controls['batch_size'] = max(1, controls.get('batch_size', 8)//2)
            db.execute('UPDATE attempts SET finished=?,state=?,result=? WHERE token=?',
                       (time.time(), state, json.dumps(result), token))
            db.execute('UPDATE jobs SET state=?,config=?,owner=NULL,lease=NULL,result=? WHERE id=?',
                       ('pending' if retry else state, json.dumps(config), json.dumps(result), job['id']))
            db.execute('DELETE FROM resources WHERE token=?', (token,))

    def recover(self, is_running):
        """Caller must establish owned-process death; expiry alone never frees GPUs."""
        recovered=[]
        with self.transaction() as db:
            for row in db.execute("SELECT * FROM jobs WHERE state='running' AND lease<?", (time.time(),)).fetchall():
                if is_running(dict(row)): continue
                state='pending' if row['attempt'] < row['max_attempts'] else 'failed'
                db.execute('UPDATE attempts SET state=?,finished=?,result=? WHERE token=?',
                           ('abandoned',time.time(),json.dumps({'reason':'owner died'}),row['owner']))
                db.execute('DELETE FROM resources WHERE token=?',(row['owner'],))
                db.execute('UPDATE jobs SET state=?,owner=NULL,lease=NULL WHERE id=?',(state,row['id']))
                recovered.append(row['id'])
        return recovered

    def rows(self):
        with self.connect() as db:return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY created')]

    def backup(self, destination):
        destination=Path(destination)
        if destination.exists():raise FileExistsError(destination)
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':raise RuntimeError('Backup integrity failed')
