from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm.decl_api import DeclarativeBase
from sqlalchemy.sql import func


Base: DeclarativeBase = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    full_name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    is_blocked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    subscription = relationship("Subscription", back_populates="user", uselist=False)
    reservations = relationship("Reservation", back_populates="user")

    def __repr__(self):
        return f"<User(user_id='{self.user_id}', email='{self.email}', is_blocked={self.is_blocked})>"


class Subscription(Base):
    __tablename__ = "subscriptions"

    subscription_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), unique=True, nullable=False)
    status = Column(String, nullable=False)
    tier = Column(String, nullable=False)
    monthly_quota = Column(Integer, nullable=False)
    started_at = Column(DateTime, default=func.now(), nullable=False)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="subscription")

    def __repr__(self):
        return f"<Subscription(subscription_id='{self.subscription_id}', user_id='{self.user_id}', status='{self.status}', tier='{self.tier}')>"


class Experience(Base):
    __tablename__ = "experiences"

    experience_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    location = Column(String, nullable=False)
    when = Column(DateTime, nullable=False)
    slots_available = Column(Integer, nullable=False)
    is_premium = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    reservations = relationship("Reservation", back_populates="experience")

    def __repr__(self):
        return f"<Experience(experience_id='{self.experience_id}', title='{self.title}', when='{self.when}')>"


class Reservation(Base):
    __tablename__ = "reservations"

    reservation_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    experience_id = Column(String, ForeignKey("experiences.experience_id"), nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="reservations")
    experience = relationship("Experience", back_populates="reservations")

    def __repr__(self):
        return f"<Reservation(reservation_id='{self.reservation_id}', user_id='{self.user_id}', experience_id='{self.experience_id}', status='{self.status}')>"

